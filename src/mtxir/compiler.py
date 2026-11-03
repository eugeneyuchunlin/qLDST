"""Compile a list of MtxIRLayer into a Stim circuit.

Each layer is one time step and is measured for ``layer.rounds`` rounds, or
``rounds_per_step`` if that is None.
A layer's checks are the paired rows of ``layer.sx`` / ``layer.sz``, so a
row may be a mixed Pauli product (e.g. a BB X check times Z on auxiliary
edge qubits after a LOAD); ``hx`` / ``hz`` are not used.

Scheduling
----------
Every check gets its own ancilla.  Pure Z checks use |0>, CX(data, ancilla)
and M; every other check uses |+>, controlled-P(ancilla, data) and MX.

* Layers that are exactly the stacked code blocks get one interleaved
  schedule in which X and Z checks run together:
    - BB codes: IBM's depth-7 CNOT order (Bravyi et al., Nature 627, 778),
      X checks use terms A2 B2 B1 B3 A1 A3 in slots 1-6 and Z checks use
      A1^T A3^T B1^T B2^T B3^T A2^T in slots 0-5.
    - Surface codes: hook-safe plaquette order.  X checks run N order
      (TL BL TR BR) so X hooks are vertical, across the horizontal X
      logical; Z checks run Z order (TL TR BL BR) so Z hooks are horizontal,
      across the vertical Z logical.
* Any other layer (e.g. the merged LOAD layer) is split into groups whose
  checks never act with different Paulis on a shared qubit, measured one
  group after another; inside a group the ancilla-data graph is edge
  colored with Konig's algorithm, with no per-check order.

Every schedule is checked for gate collisions and for correctness: two
checks that act with anticommuting Paulis on the qubit set S must have an
even number of qubits in S where the first one's gate comes first.

Noise (``p`` > 0): DEPOLARIZE2 after two-qubit gates, DEPOLARIZE1 on idle
qubits in every CNOT layer, bit flips after resets and before measurements.
Layers in ``noiseless_layers`` (indices) get no noise at all, e.g. a perfect
initialization round before a gadget; ``noiseless_readout`` does the same for
the final transversal readout.

Columns
-------
Columns are only ever appended.  New columns are reset in their initial
basis (memory blocks ``memory_basis``, surface blocks ``surface_basis``
(default X, i.e. |+>), auxiliary
qubits ``aux_basis``, a basis or a {column: basis} dict).  An auxiliary
column retires after the last layer that touches it: it is measured in
its initial basis before the next layer and left idle from then on.

Detectors
---------
* Within a layer: each check against its previous round.
* First round of a layer: a check is deterministic alone if it acts only
  on new columns with their reset Pauli, or paired with the same row of
  the previous layer if the two differ only by reset Paulis on new columns
  and by Paulis on retired columns (whose readout joins the detector).
  The remaining checks, the unpaired previous checks and the retired
  readout are combined into every deterministic product, e.g. the cycle
  checks of a gadget with the readout of their edges.
* Final transversal readout: each check, or product of checks, that is
  readable in ``readout_basis``.

Observables
-----------
Products of block logicals in ``readout_basis`` that the noiseless circuit
fixes before the final readout (single logicals first, then products),
each with the earlier measurements that fix it (Stim flows).  This covers
logicals untouched by every layer and logicals moved by a measurement,
e.g. Z_SC after a LOAD, fixed by the Z7 Z_SC outcome.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, List, Sequence

import networkx as nx
import numpy as np
import stim
from galois import GF

from .codes import BivariateBicycleCode, SurfaceCode
from .ir import CodeBlock, Instruction, MtxIRCompiler, MtxIRLayer


GF2 = GF(2)

# Pauli codes are x + 2z: 1 = X, 2 = Z, 3 = Y.
CONTROLLED_PAULI = {1: "CX", 2: "CZ", 3: "CY"}
BASIS_CODE = {"X": 1, "Z": 2}

# IBM's gross-code cycle: term index per CNOT slot, 0-2 = A1-A3 (X checks)
# or B1^T-B3^T (Z checks) on left data, 3-5 = B1-B3 or A1^T-A3^T on right.
IBM_X_ORDER = [None, 1, 4, 3, 5, 0, 2]
IBM_Z_ORDER = [3, 5, 0, 1, 2, 4, None]

# Slot of each plaquette corner (row offset, col offset).
SURFACE_X_ORDER = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
SURFACE_Z_ORDER = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}


def pauli_codes(layer: MtxIRLayer) -> np.ndarray:
    sx = np.asarray(layer.sx, dtype=np.uint8)
    sz = np.asarray(layer.sz, dtype=np.uint8)
    return sx + 2 * sz


def commuting_groups(paulis: np.ndarray) -> List[List[int]]:
    """Split nonzero rows into groups that never disagree on a qubit."""
    rows = [r for r in range(paulis.shape[0]) if paulis[r].any()]

    conflict = nx.Graph()
    conflict.add_nodes_from(rows)

    for q in range(paulis.shape[1]):
        by_pauli = defaultdict(list)
        for r in np.flatnonzero(paulis[:, q]):
            by_pauli[int(paulis[r, q])].append(int(r))

        for a, b in combinations(by_pauli.values(), 2):
            conflict.add_edges_from(product(a, b))

    coloring = nx.greedy_color(conflict, strategy="largest_first")

    groups = defaultdict(list)
    for row, color in coloring.items():
        groups[color].append(row)

    return [sorted(groups[c]) for c in sorted(groups)]


def bipartite_edge_coloring(edges: Sequence[tuple]) -> List[List[tuple]]:
    """Konig edge coloring: a bipartite graph in max-degree colors."""
    at: Dict[int, Dict[int, int]] = defaultdict(dict)

    def free_color(node):
        color = 0
        while color in at[node]:
            color += 1
        return color

    for u, v in edges:
        a = free_color(u)
        b = free_color(v)

        if a in at[v]:
            # Swap a/b along the alternating path from v; it cannot reach u.
            path = []
            node, color = v, a
            while color in at[node]:
                nxt = at[node][color]
                path.append((node, nxt, color))
                node, color = nxt, (b if color == a else a)

            for x, y, c in path:
                del at[x][c]
                del at[y][c]
            for x, y, c in path:
                swapped = b if c == a else a
                at[x][swapped] = y
                at[y][swapped] = x

        at[u][a] = v
        at[v][a] = u

    layers = defaultdict(list)
    for u, v in edges:
        color = next(c for c, w in at[u].items() if w == v)
        layers[color].append((u, v))

    return [layers[c] for c in sorted(layers)]


def bb_check_slots(code: BivariateBicycleCode):
    """IBM slot of every (local check, local qubit) pair, X then Z checks."""
    l, m = code.l, code.m
    a_terms = code.poly_a.terms
    b_terms = code.poly_b.terms
    if len(a_terms) != 3 or len(b_terms) != 3:
        raise ValueError("IBM's schedule needs three-term A and B.")

    def shift(i, term, sign=1):
        x, y = divmod(i, m)
        return ((x + sign * term[0]) % l) * m + (y + sign * term[1]) % m

    def neighbors(i, x_type):
        if x_type:
            left = [shift(i, t) for t in a_terms]
            right = [code.n_data + shift(i, t) for t in b_terms]
        else:
            left = [shift(i, t, -1) for t in b_terms]
            right = [code.n_data + shift(i, t, -1) for t in a_terms]
        return left + right

    x_slots, z_slots = [], []
    for slots, order, matrix, x_type in (
        (x_slots, IBM_X_ORDER, code.hx, True),
        (z_slots, IBM_Z_ORDER, code.hz, False),
    ):
        for i in range(matrix.shape[0]):
            nbs = neighbors(i, x_type)
            assert sorted(nbs) == np.flatnonzero(matrix[i]).tolist()
            slots.append({
                nbs[term]: slot
                for slot, term in enumerate(order)
                if term is not None
            })

    return x_slots, z_slots


def surface_check_slots(code: SurfaceCode):
    """Hook-safe slot of every (local check, local qubit) pair."""
    d = code.d

    def plaquette(support):
        rows, cols = zip(*(divmod(int(q), d) for q in support))
        if len(support) == 4:
            return min(rows), min(cols)
        if cols[0] == cols[1]:  # vertical boundary pair
            return min(rows), (d - 1 if cols[0] == d - 1 else -1)
        return (-1 if rows[0] == 0 else d - 1), min(cols)

    def slots_for(matrix, order):
        slots = []
        for row in matrix:
            support = np.flatnonzero(row)
            pr, pc = plaquette(support)
            slots.append({
                int(q): order[(int(q) // d - pr, int(q) % d - pc)]
                for q in support
            })
        return slots

    return slots_for(code.hx, SURFACE_X_ORDER), slots_for(code.hz, SURFACE_Z_ORDER)


def check_schedule(paulis: np.ndarray, time_of: Dict[int, Dict[int, int]]):
    """Assert no gate collisions and that the schedule measures each check."""
    users = defaultdict(list)
    busy = set()

    for row, times in time_of.items():
        assert sorted(times) == np.flatnonzero(paulis[row]).tolist()
        for q, t in times.items():
            assert (q, t) not in busy, f"qubit {q} used twice at time {t}"
            busy.add((q, t))
            users[q].append(row)

    parity = defaultdict(int)
    for q, rows in users.items():
        for a, b in combinations(sorted(rows), 2):
            if paulis[a, q] != paulis[b, q]:
                parity[(a, b)] ^= time_of[a][q] < time_of[b][q]

    bad = [pair for pair, odd in parity.items() if odd]
    assert not bad, f"schedule does not measure checks {bad[:5]}"


@dataclass
class Phase:
    """Checks prepared together, entangled in ``layers``, then measured."""
    rows: List[int]
    # Each layer holds (row, data qubit) pairs acting in parallel.
    layers: List[List[tuple]]


class CircuitCompiler:

    def __init__(
        self,
        memory_code_blocks: List[CodeBlock],
        surface_code_blocks: List[CodeBlock],
        *,
        rounds_per_step: int = 12,
        aux_basis: str | Dict[int, str] = "Z",
        memory_basis: str = "Z",
        readout_basis: str = "Z",
        p: float = 0.0,
        schedules: Dict[int, Dict[int, Dict[int, int]]] | None = None,
        noiseless_layers: Sequence[int] = (),
        surface_basis: str = "X",
        noiseless_readout: bool = False,
    ):
        self.memory_code_blocks = list(memory_code_blocks)
        self.surface_code_blocks = list(surface_code_blocks)
        self.rounds_per_step = rounds_per_step
        self.aux_basis = aux_basis
        self.readout_basis = readout_basis
        self.p = p
        self.noiseless_layers = set(noiseless_layers)
        self.noiseless_readout = noiseless_readout
        # Optional explicit CNOT times per layer index: row -> {qubit: time}.
        self.schedules = schedules or {}

        self.block_basis = [
            (block, memory_basis) for block in self.memory_code_blocks
        ] + [
            (block, surface_basis) for block in self.surface_code_blocks
        ]
        for block, basis in self.block_basis:
            # MtxIRCompiler chose teleport types from the blocks' reset basis.
            if getattr(block, "init_basis", None) not in (None, basis):
                raise ValueError(
                    f"Reset basis {basis!r} differs from the "
                    f"MtxIRCompiler's {block.init_basis!r}."
                )

    # ------------------------------------------------------------------
    # Circuit primitives
    # ------------------------------------------------------------------
    def _rec(self, absolute_index: int) -> stim.GateTarget:
        return stim.target_rec(absolute_index - self.n_measurements)

    def _reset(self, qubits, basis):
        if not qubits:
            return
        self.circuit.append("R" if basis == "Z" else "RX", qubits)
        if self.p:
            flip = "X_ERROR" if basis == "Z" else "Z_ERROR"
            self.circuit.append(flip, qubits, self.p)

    def _measure(self, qubits, basis) -> List[int]:
        if not qubits:
            return []
        if self.p:
            flip = "X_ERROR" if basis == "Z" else "Z_ERROR"
            self.circuit.append(flip, qubits, self.p)
        self.circuit.append("M" if basis == "Z" else "MX", qubits)

        first = self.n_measurements
        self.n_measurements += len(qubits)
        return list(range(first, self.n_measurements))

    def _detector(self, measurements, coords, support):
        self.circuit.append(
            "DETECTOR", [self._rec(m) for m in measurements], coords
        )
        self.detector_regions.append(self._region(support))

    def _region(self, support):
        """"<region>:<basis>" of a detector comparing the Pauli rows ``support``.

        Region is "sc" or "bb" if the rows act inside one such block, else
        "gadget"; basis is "X" or "Z" for pure checks, else "mixed".
        """
        owners = set(self.col_owner[np.flatnonzero(support.any(axis=0))].tolist())
        if len(owners) == 1 and -1 not in owners:
            region = self.owner_kind[owners.pop()]
        else:
            region = "gadget"

        codes = set(np.unique(support[support != 0]).tolist())
        basis = {
            BASIS_CODE["X"]: "X", BASIS_CODE["Z"]: "Z"
        }.get(codes.pop(), "mixed") if len(codes) == 1 else "mixed"
        return f"{region}:{basis}"

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------
    def _block_slots(self, n_x_rows_total):
        """Per-row slots of the plain stacked code blocks, and their Paulis."""
        n_rows = n_x_rows_total + sum(
            block.hz.shape[0] for block, _ in self.block_basis
        )
        n_cols = sum(block.hx.shape[1] for block, _ in self.block_basis)
        expected = np.zeros((n_rows, n_cols), np.uint8)
        time_of = {}

        for block, _ in self.block_basis:
            if isinstance(block.code, SurfaceCode):
                x_slots, z_slots = surface_check_slots(block.code)
            else:
                x_slots, z_slots = bb_check_slots(block.code)

            col = block.local_to_global_col_mapping
            for kind, slots, offset, code in (
                ("x", x_slots, 0, BASIS_CODE["X"]),
                ("z", z_slots, n_x_rows_total, BASIS_CODE["Z"]),
            ):
                for local, slot_of in enumerate(slots):
                    row = offset + block.local_to_global_row_mapping[kind][local]
                    time_of[row] = {col[q]: t for q, t in slot_of.items()}
                    expected[row, list(time_of[row])] = code

        return expected, time_of

    def default_times(self, paulis: np.ndarray) -> Dict[int, Dict[int, int]]:
        """CNOT time of every (row, qubit) pair.

        Rows whose block part is an original check keep the original
        hook-safe slots (IBM / surface N-Z order); a gadget edge added to a
        deformed check goes right before its first original CNOT, so its
        hooks are the original ones.  The remaining rows (vertex and cycle
        checks) follow in commuting groups, each edge-colored.
        """
        n_x_rows_total = sum(block.hx.shape[0] for block, _ in self.block_basis)
        expected, native = self._block_slots(n_x_rows_total)
        n_rows, n_cols = expected.shape

        time_of = {}
        busy = set()
        for row in range(min(n_rows, paulis.shape[0])):
            if np.array_equal(paulis[row, :n_cols], expected[row]):
                time_of[row] = dict(native[row])
                busy.update((q, t) for q, t in native[row].items())

        for row in time_of:
            for q in n_cols + np.flatnonzero(paulis[row, n_cols:]):
                t = min(time_of[row].values()) - 1
                while (int(q), t) in busy:
                    t -= 1
                time_of[row][int(q)] = t
                busy.add((int(q), t))

        rest = [r for r in range(paulis.shape[0]) if paulis[r].any() and r not in time_of]
        t0 = 1 + max((t for slots in time_of.values() for t in slots.values()), default=-1)
        if rest:
            sub = np.zeros_like(paulis)
            sub[rest] = paulis[rest]
            for rows in commuting_groups(sub):
                edges = [
                    (self.ancilla_offset + row, int(q))
                    for row in rows
                    for q in np.flatnonzero(paulis[row])
                ]
                colors = bipartite_edge_coloring(edges)
                for t, color_class in enumerate(colors):
                    for a, q in color_class:
                        time_of.setdefault(a - self.ancilla_offset, {})[int(q)] = t0 + t
                t0 += len(colors)

        return time_of

    def _schedule(self, paulis: np.ndarray, time_of=None) -> List[Phase]:
        """Phases of a round: rows whose CNOT intervals overlap share a phase."""
        if time_of is None:
            time_of = self.default_times(paulis)
        check_schedule(paulis, time_of)

        span = {
            row: (min(slots.values()), max(slots.values()))
            for row, slots in time_of.items()
        }
        phases = []
        end = None
        for row in sorted(span, key=lambda r: span[r]):
            first, last = span[row]
            if end is None or first > end:
                phases.append([])
                end = last
            phases[-1].append(row)
            end = max(end, last)

        out = []
        for rows in phases:
            start = min(span[r][0] for r in rows)
            stop = max(span[r][1] for r in rows)
            layers = [[] for _ in range(stop - start + 1)]
            for row in rows:
                for q, t in time_of[row].items():
                    layers[t - start].append((row, q))
            out.append(Phase(rows=sorted(rows), layers=layers))
        return out

    def _measure_round(self, paulis, phases: List[Phase]) -> Dict[int, int]:
        records = {}
        z_code = BASIS_CODE["Z"]
        pure_z = {
            row for row in range(paulis.shape[0])
            if paulis[row].any() and np.all(paulis[row] % z_code == 0)
        }
        data = set(range(paulis.shape[1])) - self.retired

        for phase in phases:
            ancilla = {row: self.ancilla_offset + row for row in phase.rows}
            z_rows = [r for r in phase.rows if r in pure_z]
            x_rows = [r for r in phase.rows if r not in pure_z]

            self._reset([ancilla[r] for r in z_rows], "Z")
            self._reset([ancilla[r] for r in x_rows], "X")
            self.circuit.append("TICK")

            active = set(data) | set(ancilla.values())
            for layer in phase.layers:
                by_gate = defaultdict(list)
                for row, q in layer:
                    if row in pure_z:
                        by_gate["CX"] += [q, ancilla[row]]
                    else:
                        gate = CONTROLLED_PAULI[int(paulis[row, q])]
                        by_gate[gate] += [ancilla[row], q]

                for gate, targets in by_gate.items():
                    self.circuit.append(gate, targets)
                    if self.p:
                        self.circuit.append("DEPOLARIZE2", targets, self.p)
                if self.p:
                    used = {t for targets in by_gate.values() for t in targets}
                    idle = sorted(active - used)
                    if idle:
                        self.circuit.append("DEPOLARIZE1", idle, self.p)
                self.circuit.append("TICK")

            for rows, basis in ((z_rows, "Z"), (x_rows, "X")):
                measured = self._measure([ancilla[r] for r in rows], basis)
                records.update(zip(rows, measured))

        return records

    # ------------------------------------------------------------------
    # Detectors and observables
    # ------------------------------------------------------------------
    def _first_round_detectors(
        self, paulis, previous_paulis, records, previous_records, retired,
        round_index,
    ):
        """Emit the detectors of a layer's first round.

        ``retired`` maps columns read out since the previous round to their
        measurements.  Returns the rows paired with the previous layer
        without retired readout, i.e. whose outcome carries over.
        """
        n = paulis.shape[1]
        n_old = 0 if previous_paulis is None else previous_paulis.shape[1]
        previous = (
            np.zeros((0, n), np.uint8) if previous_paulis is None
            else np.pad(previous_paulis, ((0, 0), (0, n - n_old)))
        )

        # Pauli by which a column may differ: reset Paulis of new columns,
        # readout Paulis of retired ones.
        fresh = np.zeros(n, np.uint8)
        fresh[n_old:] = self.init_code[n_old:n]
        allowed = fresh.copy()
        for q in retired:
            allowed[q] = self.init_code[q]

        leftover, paired, carried = [], set(), set()

        for row, measurement in records.items():
            if np.all((paulis[row] == 0) | (paulis[row] == fresh)):
                self._detector(
                    [measurement], [row, round_index], paulis[[row]]
                )
                continue

            if row < previous.shape[0]:
                diff = paulis[row] ^ previous[row]
                if np.all((diff == 0) | (diff == allowed)):
                    self._detector(
                        [previous_records[row], measurement]
                        + [retired[q] for q in np.flatnonzero(diff) if q in retired],
                        [row, round_index],
                        np.vstack([paulis[[row]], previous[[row]]]),
                    )
                    paired.add(row)
                    if not any(q in retired for q in np.flatnonzero(diff)):
                        carried.add(row)
                    continue

            leftover.append(row)

        # Candidates: (Pauli row, measurement or None for a reset, kind).
        members = [(paulis[row], records[row], "new") for row in leftover]
        members += [
            (previous[row], previous_records[row], "previous")
            for row in range(previous.shape[0])
            if row not in paired and previous[row].any()
        ]
        for q, measurement in retired.items():
            single = np.zeros(n, np.uint8)
            single[q] = self.init_code[q]
            members.append((single, measurement, "retired"))
        for q in range(n_old, n):
            single = np.zeros(n, np.uint8)
            single[q] = self.init_code[q]
            members.append((single, None, "reset"))

        for chosen in self._deterministic_products(members):
            self._detector(
                [members[i][1] for i in chosen if members[i][1] is not None],
                [chosen[0], round_index],
                np.vstack([members[i][0] for i in chosen]),
            )

        return carried

    def _deterministic_products(self, members):
        """Index sets of ``members`` whose Paulis multiply to the identity.

        E.g. dependent BB X checks extended by auxiliary Z factors: each is
        random, but a product cancelling the X part is Z on |0> qubits; or
        a cycle check read out through the Z readout of its edges.
        Products of previous checks and resets alone are skipped.
        """
        if not members:
            return []

        codes = np.array([pauli for pauli, _, _ in members])
        bits = GF2(np.hstack([codes & 1, codes >> 1]).astype(int))
        kinds = [kind for _, _, kind in members]

        products = []
        for combo in np.asarray(bits.left_null_space(), dtype=np.uint8):
            chosen = np.flatnonzero(combo).tolist()
            if any(kinds[i] in ("new", "retired") for i in chosen):
                products.append(chosen)

        return products

    def _readable_products(self, paulis, rows):
        """Products of ``rows`` with no Pauli outside ``readout_basis``.

        Returns (row subset, readout support) pairs.
        """
        if not rows:
            return []

        rows = np.asarray(rows)
        readout_bit = 0 if self.readout_basis == "X" else 1
        other = GF2(((paulis[rows] >> (1 - readout_bit)) & 1).astype(int))
        readout = (paulis[rows] >> readout_bit) & 1

        products = []
        for combo in np.asarray(other.left_null_space(), dtype=np.uint8):
            chosen = rows[np.flatnonzero(combo)]
            support = np.flatnonzero(readout[combo == 1].sum(axis=0) % 2)
            products.append((chosen.tolist(), support.tolist()))

        return products

    def _block_logicals(self, block: CodeBlock, basis: str):
        code = block.code
        if isinstance(code, SurfaceCode):
            supports = [code.logical_qubit_support(basis)]
        else:
            matrix = code.lz if basis == "Z" else code.lx
            supports = [
                np.flatnonzero(row).tolist()
                for row in np.asarray(matrix.todense())
            ]

        col_map = block.local_to_global_col_mapping
        return [[col_map[q] for q in support] for support in supports]

    def _observables(self, circuit, readout):
        """Measurement sets of the deterministic logicals in ``readout_basis``.

        ``circuit`` is everything before the final ``readout``.  The block
        logicals' span is intersected with the stabilizers of the noiseless
        final state; single logicals are preferred, then the basis is
        completed with products.  Stim flows give the earlier measurements
        that fix each one.
        """
        supports = [
            support
            for block, _ in self.block_basis
            for support in self._block_logicals(block, self.readout_basis)
        ]

        simulator = stim.TableauSimulator()
        simulator.do_circuit(circuit.without_noise())
        n_qubits = simulator.num_qubits

        bit = 0 if self.readout_basis == "X" else 1
        candidates = np.zeros((len(supports), 2 * n_qubits), np.uint8)
        for i, support in enumerate(supports):
            candidates[i, bit * n_qubits + np.asarray(support)] = 1

        stabilizers = np.array([
            np.hstack(s.to_numpy()) for s in simulator.canonical_stabilizers()
        ], dtype=np.uint8)

        null = GF2(np.vstack([candidates, stabilizers]).astype(int))
        null = null.left_null_space()[:, :len(supports)]
        span = GF2(np.asarray(null, dtype=int)).row_reduce()
        span = span[np.any(span != 0, axis=1)]

        def rank(rows):
            return np.linalg.matrix_rank(GF2(np.array(rows, dtype=int)))

        chosen = []
        for vector in [np.eye(len(supports), dtype=int)[i]
                       for i in range(len(supports))] + list(np.asarray(span)):
            in_span = rank(list(span) + [vector]) == len(span)
            if in_span and rank(chosen + [vector]) == len(chosen) + 1:
                chosen.append(np.asarray(vector, dtype=int))

        flows, product_supports = [], []
        for combo in chosen:
            columns = np.zeros(n_qubits, np.uint8)
            for i in np.flatnonzero(combo):
                columns[supports[i]] ^= 1
            support = np.flatnonzero(columns).tolist()

            pauli = stim.PauliString(n_qubits)
            for q in support:
                pauli[q] = self.readout_basis
            flows.append(stim.Flow(output=pauli))
            product_supports.append(support)

        noiseless = circuit.without_noise()
        solved = noiseless.solve_flow_measurements(flows)

        # Records with a fixed noiseless value only shift an observable by
        # a constant, so they are dropped (64 shots; has_flow confirms).
        shots = noiseless.compile_sampler().sample(64)
        fixed = np.all(shots == shots[0], axis=0)

        observables = []
        for flow, measurements, support in zip(flows, solved, product_supports):
            assert measurements is not None
            # A check's records agree while it carries over, so move each
            # to the check's first record; repeated pairs then cancel.
            odd = set()
            for m in measurements:
                odd ^= {self.first_record.get(self.record_row.get(m), m)}
            measurements = sorted(m for m in odd if not fixed[m])
            assert noiseless.has_flow(stim.Flow(
                output=flow.output_copy(), measurements=measurements
            ))
            observables.append(
                measurements + [readout[q] for q in support]
            )

        return observables

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------
    def compile(self, layers: List[MtxIRLayer]) -> stim.Circuit:
        all_paulis = [pauli_codes(layer) for layer in layers]
        n_max = max(p.shape[1] for p in all_paulis)
        self.ancilla_offset = n_max

        if isinstance(self.aux_basis, str):
            self.init_code = np.full(n_max, BASIS_CODE[self.aux_basis], np.uint8)
        else:
            self.init_code = np.full(n_max, BASIS_CODE["Z"], np.uint8)
            for q, basis in self.aux_basis.items():
                self.init_code[q] = BASIS_CODE[basis]
        # Block index owning each column (-1 = auxiliary), for detector regions.
        self.col_owner = np.full(n_max, -1)
        self.owner_kind = []
        for index, (block, basis) in enumerate(self.block_basis):
            for q in block.local_to_global_col_mapping.values():
                self.init_code[q] = BASIS_CODE[basis]
                self.col_owner[q] = index
            self.owner_kind.append(
                "sc" if isinstance(block.code, SurfaceCode) else "bb"
            )

        # Auxiliary columns retire after the last layer touching them.
        last_use = np.full(n_max, len(layers) - 1)
        for index, paulis in enumerate(all_paulis):
            touched = np.flatnonzero(paulis.any(axis=0))
            last_use[touched[self.col_owner[touched] == -1]] = index

        self.circuit = stim.Circuit()
        self.n_measurements = 0
        self.detector_regions = []
        self.retired = set()
        # (layer, row) of each check record, and the first record of each,
        # carried over from earlier layers while the check is unchanged.
        self.record_row = {}
        self.first_record = {}

        previous_paulis = None
        previous_records = None
        round_index = 0

        noise = self.p
        for index, (layer, paulis) in enumerate(zip(layers, all_paulis)):
            self.p = 0.0 if index in self.noiseless_layers else noise
            retired = {}
            if index:
                leaving = np.flatnonzero(last_use[:previous_paulis.shape[1]] == index - 1)
                for basis, code in BASIS_CODE.items():
                    qubits = [int(q) for q in leaving if self.init_code[q] == code]
                    retired.update(zip(qubits, self._measure(qubits, basis)))
                self.retired.update(retired)

            n_old = 0 if previous_paulis is None else previous_paulis.shape[1]
            for basis, code in BASIS_CODE.items():
                fresh = [
                    q for q in range(n_old, paulis.shape[1])
                    if self.init_code[q] == code
                ]
                self._reset(fresh, basis)

            phases = self._schedule(paulis, self.schedules.get(index))

            for r in range(layer.rounds or self.rounds_per_step):
                records = self._measure_round(paulis, phases)
                for row, measurement in records.items():
                    self.record_row[measurement] = (index, row)
                    self.first_record.setdefault((index, row), measurement)

                if r == 0:
                    carried = self._first_round_detectors(
                        paulis, previous_paulis, records,
                        previous_records, retired, round_index,
                    )
                    for row in carried:
                        self.first_record[(index, row)] = (
                            self.first_record[(index - 1, row)]
                        )
                else:
                    for row, measurement in records.items():
                        self._detector(
                            [previous_records[row], measurement],
                            [row, round_index], paulis[[row]],
                        )

                previous_records = records
                round_index += 1

            previous_paulis = paulis

        self.p = 0.0 if self.noiseless_readout else noise

        # Final transversal readout of every data qubit still in use.
        before_readout = self.circuit.copy()
        paulis = previous_paulis
        data = [q for q in range(paulis.shape[1]) if q not in self.retired]
        readout = dict(zip(data, self._measure(data, self.readout_basis)))

        for rows, support in self._readable_products(
            paulis, list(previous_records)
        ):
            self._detector(
                [previous_records[row] for row in rows]
                + [readout[q] for q in support],
                [rows[0], round_index], paulis[rows],
            )

        observables = self._observables(before_readout, readout)

        for index, measurements in enumerate(observables):
            self.circuit.append(
                "OBSERVABLE_INCLUDE",
                [self._rec(m) for m in measurements],
                index,
            )

        self.p = noise
        return self.circuit


def build_example(
    rounds_per_step: int = 12,
    p: float = 0.0,
    load_step: int = 0,
    basis: str = "Z",
    initial_placement = 1,
) -> tuple[stim.Circuit, List[str]]:
    """LOAD of q0 from one gross code block into one d=11 surface code.

    q0 sits in gross-code logical k = ``initial_placement`` (from 1).  The
    LOAD measures Z_k Z_SC for ``rounds_per_step`` rounds, runs one round
    of the original codes, measures X_k for ``rounds_per_step`` rounds and
    runs one more round of the original codes before the final readout.

    ``load_step`` is the time step of the LOAD: 0 starts directly in the
    merged code, 1 first runs one step of plain memory.  ``basis`` is the
    memory block's initial basis and the readout basis: "Z" checks the
    teleported Z information (the X^m correction), "X" the X information
    (the Z^s correction).

    Returns the circuit and each detector's "<region>:<basis>" tag, e.g.
    "sc:X", "bb:Z", "gadget:mixed".
    """
    bb = BivariateBicycleCode.generate_code(12)
    surface = SurfaceCode(11)

    bb_code_block = CodeBlock(bb)
    sc_code_block = CodeBlock(surface)

    instructions: List[Instruction] = [
        Instruction(load_step, "LOAD", "q0", "SC[0]")
    ]

    mtx_ir_compiler = MtxIRCompiler(
        initial_memory_code_blocks=[bb_code_block],
        initial_surface_code_blocks=[sc_code_block],
        memory_basis=basis,
    )
    mtx_ir_compiler.initialize("q0", True, 0, initial_placement)
    mtx_ir_compiler.compile(instructions)

    compiler = CircuitCompiler(
        [bb_code_block],
        [sc_code_block],
        rounds_per_step=rounds_per_step,
        aux_basis=mtx_ir_compiler.aux_basis,
        memory_basis=basis,
        readout_basis=basis,
        p=p,
    )
    circuit = compiler.compile(mtx_ir_compiler.MtxIRs)
    return circuit, compiler.detector_regions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=float, default=0.0)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--load-step", type=int, default=0)
    parser.add_argument("--basis", choices=["Z", "X"], default="Z")
    parser.add_argument("--out", default="mtx_ir_load_gadget_bb12_surf11.stim")
    parser.add_argument("--placement", default=1, type=int)
    args = parser.parse_args()

    circuit, _ = build_example(args.rounds, args.p, args.load_step, args.basis, args.placement)
    circuit.to_file(args.out)
    print(
        f"LOAD at step {args.load_step}, {args.rounds} rounds per step, "
        f"{args.basis} basis: "
        f"{circuit.num_qubits} qubits, {circuit.num_measurements} "
        f"measurements, {circuit.num_detectors} detectors, "
        f"{circuit.num_observables} observables -> {args.out}"
    )
