
from galois import GF, FieldArray
from typing import List, Dict

import numpy as np
import networkx as nx
import re

from dataclasses import dataclass, field
from enum import Enum
from .codes import SurfaceCode, BivariateBicycleCode
from .gf2 import block_diag_gf2, check_commute
from .polynomial import Polynomial
from itertools import combinations
from copy import deepcopy


GF2 = GF(2)

from typing import List

class MemoryInstruction(str, Enum):
    """
    A memory operation and the number of time steps it occupies.
    MemoryInstruction("LOAD") still looks a member up by its name.
    """

    LOAD = ("LOAD", 3)
    STORE = ("STORE", 3)

    def __new__(cls, name: str, time_steps: int):
        member = str.__new__(cls, name)
        member._value_ = name
        member.time_steps = time_steps
        return member



@dataclass(frozen=True)
class Instruction:
    """
    Class Instruction defines the load and store instruction. 
    The syntax is [TIME] [LOAD/STORE] [QUBIT] [DEST]. 
    For example, 
        Instruction(1, LOAD, "q0", "SC[0]")
        Instruction(5, STORE, "q1", "MEM[0][0]")

    We assume memory will be some high-rate code that stores multiple logical qubits
    in a single block, and surface code only stores one logical qubit. They are treated
    as two types of storage units, surface code as an 1D array composed of several
    patches and memory blocks (memory banks) as a 2D array.
    Unlike qubit, we view the surface code patch and memory block as nameless units.
    """


    time_index: int
    instr: MemoryInstruction | str
    qubit: str
    destination: str

    destination_prefix: str = field(init=False)
    block_index: int = field(init=False)
    slot_index: int | None = field(init=False)

    def __post_init__(self):
        operation = MemoryInstruction(self.instr)
        object.__setattr__(self, "instr", operation)

        match = re.fullmatch(
            r"(SC|MEM)\[(\d+)\](?:\[(\d+)\])?",
            self.destination,
        )
        if match is None:
            raise ValueError(f"Invalid destination: {self.destination!r}")

        prefix, block, slot = match.groups()

        if prefix == "SC" and slot is not None:
            raise ValueError("Use SC[block].")

        if prefix == "MEM" and slot is None:
            raise ValueError("Use MEM[block][slot].")

        expected = "SC" if operation == MemoryInstruction.LOAD else "MEM"
        if prefix != expected:
            raise ValueError(
                f"{operation.value} requires a {expected} destination."
            )

        object.__setattr__(self, "destination_prefix", prefix)
        object.__setattr__(self, "block_index", int(block))
        object.__setattr__(
            self, "slot_index", int(slot) if slot is not None else None
        )

class CodeBlockType(Enum):
    MEM = "MEM"
    SC = "SC"

class CodeBlock:
    """High-level storage of logical qubits."""

    def __init__(self, code, type: CodeBlockType | None = None):
        """``type`` defaults to SC for a surface code and MEM otherwise."""
        self.hx = code.hx.copy()
        self.hz = code.hz.copy()
        self.capacity = code.k
        self.size = 0
        self.qubits = [None] * self.capacity
        self.code = code
        self.local_to_global_row_mapping = {}
        self.local_to_global_col_mapping = {}
        # Reset basis of the data qubits; set by MtxIRCompiler.
        self.init_basis = None
        # Per slot: None if occupied, else (basis, sign) of the eigenstate
        # the empty slot is in.  sign is None for +1, or (time step, vertex
        # check rows) of the measurement whose outcome gives it.
        self.empty_state = [None] * self.capacity
        if type is None:
            type = (CodeBlockType.SC if isinstance(code, SurfaceCode)
                    else CodeBlockType.MEM)
        self.type: CodeBlockType = type

    def first_empty_spot(self):
        for i, q in enumerate(self.qubits):
            if q == None:
                return i

    def locate(self, qubit: str):
        """Slot of ``qubit``, counting from 0."""
        return self.qubits.index(qubit)



class MtxIRLayer:
    """One time step's checks.

    ``rounds`` is the number of syndrome rounds; None leaves it to the
    circuit compiler's default.
    """

    def __init__(
        self, time_index, hx=None, hz=None, *, sx=None, sz=None, rounds=None
    ):
        self.time_step = time_index
        self.rounds = rounds

        if sx is None and sz is None:
            if hx is None or hz is None:
                raise ValueError("Provide hx/hz or sx/sz.")

            check_commute(hx, hz)

            self.hx = hx.copy()
            self.hz = hz.copy()

            n = hx.shape[1]
            rx = hx.shape[0]
            rz = hz.shape[0]

            # Each paired row now describes the SAME check.
            self.sx = GF2.Zeros((rx + rz, n))
            self.sz = GF2.Zeros((rx + rz, n))

            self.sx[:rx] = hx
            self.sz[rx:] = hz

        else:
            if hx is not None or hz is not None:
                raise ValueError("Use only one input representation.")
            if sx is None or sz is None:
                raise ValueError("Both sx and sz are required.")

            self.hx = None
            self.hz = None
            self.sx = GF2(sx).copy()
            self.sz = GF2(sz).copy()

        assert self.sx.shape == self.sz.shape
        assert np.all(
            self.sx @ self.sz.T + self.sz @ self.sx.T == 0
        )

    @property
    def n(self):
        return self.sx.shape[1]

    def update_layer(self, hx=None, hz=None, *, sx=None, sz=None, rounds=None):
        """Replace this layer's checks, keeping its time step; ``rounds``
        replaces the round count when given."""
        replacement = MtxIRLayer(self.time_step, hx, hz, sx=sx, sz=sz,
                                 rounds=self.rounds if rounds is None else rounds)
        self.__dict__.update(replacement.__dict__)


    @property
    def qubit_overhead(self):
        return self.n + self.sx.shape[0]




class MtxIRCompiler:
    """Each layer represents one time_step."""

    def __init__(
        self,
        initial_memory_code_blocks: List[CodeBlock],
        initial_surface_code_blocks: List[CodeBlock],
        memory_basis: str = "Z",
        surface_basis: str = "X",
    ):
        """``memory_basis`` and ``surface_basis`` are the reset bases of
        the memory and surface blocks' data qubits (all their logicals start
        in |0> or |+>); they must match the circuit compiler's.
        """
        # Copies the dictionaries, not the CodeBlock objects.
        self.memory_code_blocks = list(initial_memory_code_blocks)
        self.surface_code_blocks = list(initial_surface_code_blocks)

        for blocks, basis in ((self.memory_code_blocks, memory_basis),
                              (self.surface_code_blocks, surface_basis)):
            for block in blocks:
                block.init_basis = basis
                block.empty_state = [
                    (basis, None) if q is None else None for q in block.qubits
                ]

        self.code_blocks = self.memory_code_blocks + self.surface_code_blocks
        

        hx, hz = self.stack_code_blocks(self.code_blocks)

        self.layers = [MtxIRLayer(0, hx, hz)]
        self.n = hx.shape[1]
        self.time = 1


        self.locations: Dict[str, CodeBlock] = {}

        # Reset/readout basis of every auxiliary column.
        self.aux_basis: Dict[int, str] = {}

        # One entry per LOAD or STORE; see ``_teleport``.
        self.pauli_frame: List[dict] = []


    def stack_code_blocks(
        self,
        code_blocks: List[CodeBlock],
        *,
        col_offset=0,
        x_row_offset=0,
        z_row_offset=0,
    ):
        code_blocks = list(code_blocks)

        # Validate before changing any mappings.
        for block in code_blocks:
            if block.hx.shape[1] != block.hz.shape[1]:
                raise ValueError("Hx and Hz must have the same qubit count.")
            check_commute(block.hx, block.hz)

        hx = block_diag_gf2([block.hx for block in code_blocks])
        hz = block_diag_gf2([block.hz for block in code_blocks])

        for block in code_blocks:
            n_qubits = block.hx.shape[1]
            n_x_checks = block.hx.shape[0]
            n_z_checks = block.hz.shape[0]

            block.local_to_global_col_mapping = {
                local: col_offset + local
                for local in range(n_qubits)
            }

            block.local_to_global_row_mapping = {
                "x": {
                    local: x_row_offset + local
                    for local in range(n_x_checks)
                },
                "z": {
                    local: z_row_offset + local
                    for local in range(n_z_checks)
                },
            }

            col_offset += n_qubits
            x_row_offset += n_x_checks
            z_row_offset += n_z_checks

        return hx, hz

    @property
    def hx(self):
        return self.layers[-1].hx

    @property
    def hz(self):
        return self.layers[-1].hz

    def add_new_code_blocks(self, code_blocks, in_time=True):
        if not code_blocks:
            return

        previous = self.layers[-1]

        added_hx, added_hz = self.stack_code_blocks(
            code_blocks,
            col_offset=previous.hx.shape[1],
            x_row_offset=previous.hx.shape[0],
            z_row_offset=previous.hz.shape[0],
        )

        hx = block_diag_gf2([previous.hx, added_hx])
        hz = block_diag_gf2([previous.hz, added_hz])

        check_commute(hx, hz)

        if in_time:
            previous.update_layer(hx, hz)
        else:
            self.layers.append(MtxIRLayer(self.time, hx, hz))
            self.time += 1

    def _gross_logicals(self, bb, index):
        """Local supports of Z[index] and X[index] of the gross code.

        Tour de gross Eq. (32) in this coordinate convention: L terms are
        shifted by y and R terms by x relative to the paper.
        """


        def poly(terms):

            return Polynomial(terms, orders=[12, 6])


        p = poly("x^4 + x^5 + x^6y + x^4y^2 + x^5y^4 + x^6y^5")
        q = poly("x^3 + x^4 + x^3y + x^3y^2 + x^4y^2 + x^3y^5")
        r = poly("1 + x^8 + xy + x^9y + x^3y^4 + x^11y^4")
        s = poly("x + x^9 + x^4y^4 + x^8y^4 + y^5 + x^8y^5")
        left_offset, right_offset = poly("y"), poly("x")
        shift = poly("xy")

        alpha_list = [
            poly(t)
            for t in ("1", "x^3y^5", "x^11y^5", "x^10y", "x^5y^4", "x^4y^2")
        ]
        beta_list = [
            poly(t)
            for t in ("1", "x^2y^4", "xy^2", "x^2y^5", "xy", "x^3y")
        ]


        if (bb.l, bb.m) != (12, 6):
            raise ValueError("This logical support is specific to gross code.")

        def poly(terms):
            return Polynomial(terms, orders=[12, 6])


        def native_support(left, right):
            # conver paper's polynomials into native polynomial
            return sorted(bb.support(left * left_offset, right * right_offset))

        
        assert 0 <= index < 12, "Index must be in 0..11"


        if index < 6:
            j = index
            alpha = alpha_list[j]
            beta = beta_list[j]

            x = native_support(alpha * p, alpha * q)
            z = native_support(beta * shift * s.T, beta * shift * r.T)
        else:
            j = index - 6
            alpha_T = alpha_list[j].T
            beta_T = beta_list[j].T

            x = native_support(beta_T * r, beta_T * s)
            z = native_support(alpha_T * shift * q.T, alpha_T * shift * p.T)


        for support in (z, x):
            if len(set(support)) != 12:
                raise ValueError("Expected 12 distinct logical support qubits.")

        return z, x

    def _gauging_layer(
        self,
        base: MtxIRLayer,
        block: CodeBlock,
        support: List[int],
        basis: str,
        time_index: int,
        attach=None,
    ):
        """
        Gauging measurement [WY24] of the ``basis`` logical on ``support``.

        Vertices are the (local) support qubits of ``block``; an edge joins
        two of them that share a check of the other type and carries a new
        auxiliary column, reset and read out in ``basis``. Vertex checks are
        ``basis`` on the vertex qubit times the other Pauli on its edges,
        cycle checks are ``basis`` on the edges of a cycle basis, and each
        other-type check gains ``basis`` on the edge joining its overlap.

        ``attach`` = (block, local support) joins a ``basis`` logical of a
        second block: its qubits are spread over a path of the graph, so
        the product of all vertex checks is the joint logical.

        Returns the new layer and gadget metadata.
        """
        other = "X" if basis == "Z" else "Z"
        operands = [(block, support)] + ([attach] if attach else [])
        n_x_rows = sum(b.hx.shape[0] for b in self.code_blocks)

        def other_checks(b):
            return b.hx if other == "X" else b.hz

        def global_row(b, local):
            row = b.local_to_global_row_mapping[other.lower()][local]
            return row if other == "X" else n_x_rows + row

        # New edges go after every column allocated so far, including those
        # of gaugings built after ``base`` was (e.g. the joint step before
        # the source step, or a parallel instruction).
        if base.n < self.n:
            base = self._restored(base, base.time_step, base.rounds)

        old_n = base.n

        # ------------------------------------------------------------
        # 1. Operand checks of the other type must be unmodified.
        # ------------------------------------------------------------
        base_other, base_same = (
            (base.sx, base.sz) if other == "X" else (base.sz, base.sx)
        )

        for b, operand_support in operands:
            col_map = b.local_to_global_col_mapping
            for local, row in enumerate(other_checks(b)):

                if not np.any(row[list(map(int, operand_support))]):
                    continue

                expected = GF2.Zeros(old_n)
                expected[[col_map[int(q)] for q in np.flatnonzero(row)]] = 1
                data = [col_map[q] for q in range(b.hx.shape[1])] 
                g = global_row(b, local)
                if (
                    not np.array_equal(base_other[g], expected)
                    or np.any(base_same[g][data] != 0)
                ):
                    raise ValueError(
                        f"Operand {other} checks are already modified; "
                        "restore the block before activating this gadget."
                    )

        # ------------------------------------------------------------
        # 2. Build edges using LOCAL indices.
        # ------------------------------------------------------------
        edges = []
        edge_to_check = []

        for check_id, row in enumerate(other_checks(block)):
            overlap = [q for q in support if int(row[q])]

            if overlap:
                assert len(overlap) == 2
                edges.append(tuple(sorted(overlap)))
                edge_to_check.append(check_id)

        assert len(set(edges)) == len(edges)

        graph = nx.Graph()
        graph.add_nodes_from(support)
        graph.add_edges_from(edges)

        assert nx.is_connected(graph)

        # New auxiliary columns are appended after ALL existing qubits.
        new_n = old_n + len(edges)

        edge_to_aux = {
            edge: old_n + edge_id
            for edge_id, edge in enumerate(edges)
        }

        # ------------------------------------------------------------
        # 3. Choose a path for the attached logical.
        # ------------------------------------------------------------
        attach_to_vertex = {}
        path = None

        if attach:
            attach_block, attach_support = attach

            def find_path_of_length(n_vertices):
                def search(path):
                    if len(path) == n_vertices:
                        return path

                    for neighbor in sorted(graph.neighbors(path[-1])):
                        if neighbor not in path:
                            result = search(path + [neighbor])
                            if result is not None:
                                return result

                    return None

                for start in sorted(graph.nodes):
                    result = search([start])
                    if result is not None:
                        return result

                return None

            for length in range(
                min(len(attach_support), graph.number_of_nodes()), 0, -1
            ):
                path = find_path_of_length(length)
                if path is not None:
                    break

            assert path is not None

            # Consecutive groups support strings longer than the chosen path.
            groups = np.array_split(attach_support, len(path))

            attach_to_vertex = {
                int(q): vertex
                for vertex, group in zip(path, groups)
                for q in group
            }

        # ------------------------------------------------------------
        # 4. Preserve ALL existing checks, padding auxiliary columns.
        # ------------------------------------------------------------
        sx = GF2.Zeros((base.sx.shape[0], new_n))
        sz = GF2.Zeros((base.sz.shape[0], new_n))

        sx[:, :old_n] = base.sx
        sz[:, :old_n] = base.sz

        # Rows' Pauli ``basis`` part.
        same = sz if basis == "Z" else sx

        # ------------------------------------------------------------
        # 5. Extend the operand's other-type checks with edge factors.
        # ------------------------------------------------------------
        for edge, local_check in zip(edges, edge_to_check):
            same[global_row(block, local_check), edge_to_aux[edge]] += GF2(1)

        if attach:
            for local_check, row in enumerate(other_checks(attach_block)):
                # A vertex contributes only if this check touches its
                # attached qubits an odd number of times.
                odd_vertices = set()

                for q in attach_support:
                    if int(row[q]):
                        odd_vertices ^= {attach_to_vertex[int(q)]}

                endpoints = sorted(odd_vertices)
                assert len(endpoints) % 2 == 0

                g = global_row(attach_block, local_check)

                # Join endpoint pairs; overlapping path edges cancel in GF(2).
                for a, b in zip(endpoints[::2], endpoints[1::2]):
                    route = nx.shortest_path(graph, a, b)

                    for u, v in zip(route, route[1:]):
                        edge = tuple(sorted((u, v)))
                        same[g, edge_to_aux[edge]] += GF2(1)

        # ------------------------------------------------------------
        # 6. Build vertex checks.
        # ------------------------------------------------------------
        vertex_to_row = {
            data: row
            for row, data in enumerate(support)
        }

        vertex_same = GF2.Zeros((len(support), new_n))
        vertex_other = GF2.Zeros((len(support), new_n))

        col_map = block.local_to_global_col_mapping

        for local_data, vertex_row in vertex_to_row.items():
            vertex_same[vertex_row, col_map[local_data]] = 1

            for edge in edges:
                if local_data in edge:
                    vertex_other[vertex_row, edge_to_aux[edge]] = 1

        if attach:
            attach_col_map = attach_block.local_to_global_col_mapping
            for q, vertex in attach_to_vertex.items():
                vertex_same[vertex_to_row[vertex], attach_col_map[q]] = 1

        # ------------------------------------------------------------
        # 7. Add cycle checks.
        # ------------------------------------------------------------
        # A product of other-type operand checks that is trivial on the
        # data leaves a cycle on the edges, already a stabilizer of the
        # deformed code [WY24]; only cycles outside that span need checks.
        aux_cols = list(range(old_n, new_n))
        operand_rows = [
            global_row(b, local)
            for b, _ in operands
            for local in range(other_checks(b).shape[0])
        ]
        redundancies = GF2(
            np.asarray(base_other[operand_rows], dtype=int)
        ).left_null_space()
        n_data = sum(b.hx.shape[1] for b in self.code_blocks)
        old_edges = same[operand_rows][:, n_data:old_n]
        if redundancies.shape[0] and old_edges.shape[1]:
            residue = redundancies @ old_edges
            if np.any(residue):
                redundancies = residue.left_null_space() @ redundancies

        implied = redundancies @ same[operand_rows][:, aux_cols]

        def cycle_vector(vertices):
            vector = GF2.Zeros(len(edges))
            for pos, u in enumerate(vertices):
                v = vertices[(pos + 1) % len(vertices)]
                vector[edge_to_aux[tuple(sorted((u, v)))] - old_n] = 1
            return vector

        # Greedy by length: the lightest cycles completing the span.
        longest = max([len(c) for c in nx.minimum_cycle_basis(graph)], default=0)
        candidates = sorted(
            nx.simple_cycles(graph, length_bound=longest),
            key=lambda c: (len(c), sorted(c)),
        )

        cycles = []
        span = [row for row in implied]
        rank = np.linalg.matrix_rank(GF2(np.array(span, dtype=int))) if span else 0

        for vertices in candidates:
            vector = cycle_vector(vertices)
            new_rank = np.linalg.matrix_rank(GF2(np.array(span + [vector], dtype=int)))
            if new_rank > rank:
                cycles.append(vertices)
                span.append(vector)
                rank = new_rank

        assert rank == len(edges) - len(support) + 1

        cycle_same = GF2.Zeros((len(cycles), new_n))

        for cycle_id, vertices in enumerate(cycles):
            cycle_same[cycle_id, aux_cols] = cycle_vector(vertices)

        cycle_other = GF2.Zeros((len(cycles), new_n))

        # Existing rows keep their positions; new rows are appended.
        vertex_row_start = sx.shape[0]

        if basis == "Z":
            joint_sx = GF2(np.vstack([sx, vertex_other, cycle_other]))
            joint_sz = GF2(np.vstack([sz, vertex_same, cycle_same]))
        else:
            joint_sx = GF2(np.vstack([sx, vertex_same, cycle_same]))
            joint_sz = GF2(np.vstack([sz, vertex_other, cycle_other]))

        # ------------------------------------------------------------
        # 8. Validate the measurement configuration.
        # ------------------------------------------------------------
        assert np.all(
            joint_sx @ joint_sz.T + joint_sz @ joint_sx.T == 0
        )

        expected = GF2.Zeros(new_n)

        for q in support:
            expected[col_map[q]] = 1

        if attach:
            for q in attach_support:
                expected[attach_col_map[int(q)]] = 1

        assert np.all(vertex_other.sum(axis=0) == 0)
        assert np.array_equal(vertex_same.sum(axis=0), expected)

        old_H = GF2(np.hstack([base.sx, base.sz]))
        joint_H = GF2(np.hstack([joint_sx, joint_sz]))

        old_k = old_n - np.linalg.matrix_rank(old_H)
        new_k = new_n - np.linalg.matrix_rank(joint_H)

        assert new_k == old_k - 1



        # ------------------------------------------------------------
        # 9. Register the auxiliary columns and return the layer.
        # ------------------------------------------------------------
        for aux in edge_to_aux.values():
            self.aux_basis[aux] = basis

        self.n = new_n

        layer = MtxIRLayer(time_index, sx=joint_sx, sz=joint_sz)

        return layer, {
            "basis": basis,
            "auxiliary_columns": list(range(old_n, new_n)),
            "edge_to_aux": edge_to_aux,
            "vertex_to_row": vertex_to_row,
            "attach_to_vertex": attach_to_vertex,
            "path": path,
            "vertex_check_rows": list(range(
                vertex_row_start,
                vertex_row_start + len(support),
            )),
            "measured_support": np.flatnonzero(expected).tolist(),
            "cycles": cycles,
        }

    def _restored(self, base: MtxIRLayer, time_index: int, rounds=None):
        """``base``'s checks on all current columns.

        Columns that ``base`` does not touch are idle; the circuit compiler
        reads out auxiliary columns once no later layer uses them.
        """
        def pad(matrix):
            return GF2(np.hstack([
                matrix, GF2.Zeros((matrix.shape[0], self.n - base.n))
            ]))

        if base.hx is not None:
            return MtxIRLayer(
                time_index, pad(base.hx), pad(base.hz), rounds=rounds
            )

        return MtxIRLayer(
            time_index, sx=pad(base.sx), sz=pad(base.sz), rounds=rounds
        )

    def _slot_logicals(self, block: CodeBlock, slot: int):
        """Local supports {"Z": ..., "X": ...} of logical ``slot`` (from 0)."""
        if any(block is b for b in self.memory_code_blocks):
            z, x = self._gross_logicals(block.code, slot)
        else:
            z = block.code.logical_qubit_support("Z")
            x = block.code.logical_qubit_support("X")
        return {"Z": z, "X": x}

    def _teleport(
        self,
        bases: List[MtxIRLayer],
        source: CodeBlock,
        source_slot: int,
        dest: CodeBlock,
        dest_slot: int,
        in_time=False,
    ):
        """
        Teleport the logical in ``source_slot`` of ``source`` into the empty
        ``dest_slot`` of ``dest`` (slots count from 0).  The destination is
        in an eigenstate of some basis B (see ``CodeBlock.empty_state``);
        the joint measurement uses the other basis J, so no preparation
        is needed.  There are four time steps:

          t    : measure J_source * J_dest (outcome m) for C rounds,
          t + 1: one round of the original codes; J edges read out in J,
          t + 2: measure B_source (outcome s) for C rounds,
          t + 3: one round of the original codes; B edges read out in B.

        B = X (dest in |+>): J = Z, and dest holds Z^s X^m |psi>.
        B = Z (dest in |0>): J = X, and dest holds X^s Z^m |psi>.
        If dest was in the -1 eigenstate, a further J correction applies,
        given by the outcome that fixed that sign.  The source slot is left
        empty in a B eigenstate of sign s.

        The frame entry names the vertex check rows whose first-round
        product gives each outcome.  Gauging byproducts on the edges [WY24]
        are left to the decoder: the circuit's detectors and observables
        include the edge readout.

        Assumes the original check rows retain their initial global
        positions, and that ``base`` is the unmodified code.
        """

        assert (len(bases) >= 3), "len(bases) must be at least 3"

        qubit = source.qubits[source_slot]
        if qubit is None:
            raise ValueError(f"Source slot {source_slot} is empty.")
        if dest.qubits[dest_slot] is not None:
            raise ValueError(f"Destination slot {dest_slot} is occupied.")
        state = dest.empty_state[dest_slot]
        if state is None:
            raise ValueError(
                f"Destination slot {dest_slot} is not in a known eigenstate."
            )
        dest_basis, dest_sign = state
        joint = "Z" if dest_basis == "X" else "X"

        base = bases[0]
        start = base.time_step if in_time else self.time

        source_logicals = self._slot_logicals(source, source_slot)
        dest_logicals = self._slot_logicals(dest, dest_slot)

        # The joint measurement's graph is built on the memory block, the
        # other block attached to it: a surface-code logical gives a path
        # graph, along which the attached block's checks would be extended
        # (weights up to 17 at d = 15, against 7).
        operands = [(source, source_logicals), (dest, dest_logicals)]
        if not any(source is b for b in self.memory_code_blocks):
            operands.reverse()
        (graph_block, graph_logicals), (attach_block, attach_logicals) = operands

        joint_layer, joint_gadget = self._gauging_layer(
            base, graph_block, graph_logicals[joint], joint, start,
            attach=(attach_block, attach_logicals[joint]),
        )

        updated_layers = []
        updated_layers.append(joint_layer)

        source_layer, source_gadget = self._gauging_layer(
            bases[1], source, source_logicals[dest_basis], dest_basis,
            bases[1].time_step,
        )
        updated_layers.append(source_layer)
        updated_layers.append(self._restored(bases[2], bases[2].time_step, 1))

        # ------------------------------------------------------------
        # Pauli frame and logical locations.
        # ------------------------------------------------------------
        # (Pauli, time step, vertex check rows giving its condition)
        corrections = [
            (dest_basis, start, joint_gadget["vertex_check_rows"]),
            (joint, bases[1].time_step, source_gadget["vertex_check_rows"]),
        ]
        if dest_sign is not None:
            corrections.append((joint, *dest_sign))

        self.pauli_frame.append({
            "qubit": qubit,
            "block": dest,
            "joint": joint,
            "corrections": corrections,
            "gadgets": [joint_gadget, source_gadget],
        })

        source.qubits[source_slot] = None
        source.empty_state[source_slot] = (
            dest_basis, (bases[1].time_step, source_gadget["vertex_check_rows"])
        )
        dest.qubits[dest_slot] = qubit
        dest.empty_state[dest_slot] = None
        self.locations[qubit] = dest

        return joint_gadget, source_gadget, updated_layers

    def _load(
        self,
        bases: List[MtxIRLayer],
        memory_code_block: CodeBlock,
        surface_code_block: CodeBlock,
        qubit: str,
        in_time=False,
    ):
        """Teleport ``qubit`` from a gross code block into an empty surface
        code block; see ``_teleport``.  From a fresh surface block (|+>)
        this measures Z[slot]_BB * Z_surface, then X[slot]_BB.
        """
        if surface_code_block.qubits[0] is not None:
            raise ValueError("The destination surface code block is occupied.")

        return self._teleport(bases, memory_code_block,
            memory_code_block.locate(qubit),
            surface_code_block, 0, in_time,
        )

    def _store(
        self,
        bases: List[MtxIRLayer],
        surface_code_block: CodeBlock,
        memory_code_block: CodeBlock,
        dest: int = None,
        in_time=False,
    ):
        """Teleport the logical in a surface code block into logical
        ``dest`` (from 0) of a gross code block, or into its first empty
        slot if ``dest`` is None; see ``_teleport``.  The joint measurement
        is ZZ if the slot is in an X eigenstate (e.g. freed by a LOAD) and
        XX if it is in a Z eigenstate (e.g. a fresh block reset in |0>).
        """
        if dest is None:
            dest = memory_code_block.first_empty_spot()
            if dest is None:
                raise ValueError("Memory Block is full")

        return self._teleport(bases, 
            surface_code_block, 0, memory_code_block, dest, in_time,
        )

    def initialize(self, qubit: str, is_memory_block: bool, i: int, j: int=None):
        """Place ``qubit`` in logical ``j`` (from 0) of memory block ``i``,
        or in surface block ``i``."""
        if is_memory_block:
            block = self.memory_code_blocks[i]
            if j is None or not 0 <= j < block.capacity:
                raise ValueError(
                    f"Logical index must be in 0..{block.capacity - 1}, got {j}."
                )
        else:
            block, j = self.surface_code_blocks[i], 0

        if block.qubits[j] is not None:
            raise ValueError(f"Slot {j} already holds {block.qubits[j]}.")

        block.qubits[j] = qubit
        block.empty_state[j] = None
        self.locations[qubit] = block


    def step(self):
        layer = deepcopy(self.layers[-1])
        layer.time_step += 1
        layer.rounds = None
        self.layers.append(layer)
        self.time = layer.time_step + 1

    def find_layer(self, t):
        for layer in self.layers:
            if layer.time_step == t:
                return layer
        


    def _location_name(self, qubit: str):
        """Where ``qubit`` is now, spelled like ``Instruction.destination``."""
        block = self.locations[qubit]
        if block.type == CodeBlockType.MEM:
            index = self.memory_code_blocks.index(block)
            return f"MEM[{index}][{block.locate(qubit)}]"
        return f"SC[{self.surface_code_blocks.index(block)}]"

    def compile(self, instructions: List[Instruction]):

        # check dependency
        
        occupancy: Dict[int, List[str]] = {}
        timing_groups: Dict[int, List[Instruction]] = {}

        for instr in instructions:
            if not instr.time_index in timing_groups:
                timing_groups[instr.time_index] = []

        current = {qubit: self._location_name(qubit) for qubit in self.locations}
        sources = {}
        for instr in sorted(instructions, key=lambda i: i.time_index):
            if instr.qubit not in current:
                raise ValueError(f"{instr.qubit} is not initialized.")
            sources[instr] = current[instr.qubit]
            current[instr.qubit] = instr.destination

        for instr in instructions:
            timing_groups[instr.time_index].append(instr)
            for t in range(instr.time_index, instr.time_index + instr.instr.time_steps):
                if t not in occupancy:
                    occupancy[t] = []
                occupancy[t].append(instr.qubit)
                occupancy[t].append(instr.destination)
                occupancy[t].append(sources[instr])


#         import json
#         print(json.dumps(occupancy, indent=4)) 

        for t, values in occupancy.items():
            if len(values) != len(set(values)):
                raise ValueError(f"At time {t}, qubits or blocks are involved in two operations")


        for t, instr_group in timing_groups.items():
            while self.layers[-1].time_step < t: # include the layer we are building. Later, use in-time to update the layer
                self.step()

            for instr in instr_group:
                # print("Instr: ", instr)
                # Build the layers the instruction occupies, if not yet built.
                steps = instr.instr.time_steps
                tt = 0
                while tt < t + steps:
                    if self.find_layer(tt) == None:
                        self.step()
                    tt += 1

                bases = [self.find_layer(t + k) for k in range(steps)]
                if instr.instr == MemoryInstruction.LOAD:
                    source = self.locations[instr.qubit]
                    if not any(source is b for b in self.memory_code_blocks):
                        raise ValueError(f"{instr.qubit} is not in a memory block.")

                    _, __, updated_layers = self._load(
                        bases, source, self.surface_code_blocks[instr.block_index],
                        instr.qubit, in_time=True,
                    )
                elif instr.instr == MemoryInstruction.STORE:
                    source = self.locations[instr.qubit]
                    if not any(source is sc for sc in self.surface_code_blocks):
                        raise ValueError(f"{instr.qubit} is not in a surface code block")

                    _, __, updated_layers = self._store(
                        bases, source, self.memory_code_blocks[instr.block_index],
                        instr.slot_index, in_time=True
                    )
                else:
                    raise ValueError(f"Instruction {instr.instr} is not supported yet")

                for i in range(len(bases)):
                    new = updated_layers[i]
                    if new.hx is not None:
                        bases[i].update_layer(new.hx, new.hz, rounds=new.rounds)
                    else:
                        bases[i].update_layer(sx=new.sx, sz=new.sz, rounds=new.rounds)
                


    @property
    def MtxIRs(self):
        return self.layers



if __name__ == '__main__':
    bb = BivariateBicycleCode.generate_code(12)
    surface = SurfaceCode(11)


    bb_code_block1 = CodeBlock(bb, type=CodeBlockType.MEM)
    bb_code_block2 = CodeBlock(bb, type=CodeBlockType.MEM)
    sc_code_block = CodeBlock(surface, type=CodeBlockType.SC)
    new_sc_code_block = CodeBlock(surface, type=CodeBlockType.SC)


    instructions : List[Instruction] = [
        Instruction(1, "LOAD", "q0", "SC[0]"),
        Instruction(5, "STORE", "q0", "MEM[0][0]")
    ]

    mtx_ir_compiler = MtxIRCompiler(
            initial_memory_code_blocks=[bb_code_block1], 
            initial_surface_code_blocks=[sc_code_block]
            )

    mtx_ir_compiler.initialize("q0", True, 0, 0)
    mtx_ir_compiler.compile(instructions)
    print(mtx_ir_compiler.MtxIRs[-1].qubit_overhead)