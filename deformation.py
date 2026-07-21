import enum
import numpy as np
import sympy as sp
import stim
import sinter
from galois import GF, GF2
from bposd.css import css_code
from error import NoiseModel, DepolarizingNoise, FlipNoise
from gate_profile import GateProfile
import scipy.sparse as sparse


_X, _Y = sp.symbols("x y")


class Polynomial:
    def __init__(self, polynomial_str: str = None, terms: list[tuple[int, int]] = None):
        if terms is not None:
            if polynomial_str is not None:
                raise ValueError(
                    "Polynomial accepts either polynomial_str or terms, not both."
                )
            self.terms = list(terms)
        elif polynomial_str is not None:
            self.terms = self._parse_polynomial(polynomial_str)
        else:
            raise ValueError("Polynomial requires either polynomial_str or terms.")

    def _parse_polynomial(self, polynomial_str: str):
        cleaned = polynomial_str.replace(" ", "").replace("^", "**")
        expr = sp.sympify(cleaned, evaluate=False)

        # The zero polynomial has no terms at all -- without this check,
        # sympify("0") produces sp.Zero(), which is not an Add, so it falls
        # into the `(expr,)` branch below and as_powers_dict() on it returns
        # {0: 1} (no x/y key), silently parsed as the term (0, 0) i.e. the
        # constant monomial "1". Polynomial("0") would then be identical to
        # Polynomial("1") instead of representing zero.
        if expr == 0:
            return []

        monomials = expr.args if isinstance(expr, sp.Add) else (expr,)

        results = []
        for m in monomials:
            powers = m.as_powers_dict()
            x_exp = int(powers.get(_X, 0))
            y_exp = int(powers.get(_Y, 0))
            results.append((x_exp, y_exp))

        return results

    def __repr__(self):
        return f"Polynomial(terms={self.terms})"
    

class BivariateBicycleCode:

    def __init__(self, l, m, poly_a: Polynomial, poly_b: Polynomial, d: int = None):
        self.l = l
        self.m = m
        self.n_data = l * m

        self.poly_a = poly_a
        self.poly_b = poly_b
        self.A = self._poly_to_matrix(poly_a)
        self.B = self._poly_to_matrix(poly_b)

        GF2 = GF(2)
        self.hx = GF2(np.hstack((self.A, self.B)).astype(int))
        self.hz = GF2(np.hstack((self.B.T, self.A.T)).astype(int))

        qcode = css_code(hx=self.hx, hz=self.hz)
        self.lx = qcode.lx
        self.lz = qcode.lz

        rank_hx = np.linalg.matrix_rank(self.hx)
        rank_hz = np.linalg.matrix_rank(self.hz)
        self.n = 2 * self.l * self.m
        self.k = self.n - rank_hx - rank_hz
        self.d = d # distance

        # print(f"rank_hx: {rank_hx}, rank_hz: {rank_hz}, k: {self.k}")

        assert self._check_commutation(
            self.hx, self.hz
        ), "The provided polynomials do not make commutative matrices, invalid CSS code."

    def _poly_to_matrix(self, poly: Polynomial):
        matrix = np.zeros((self.n_data, self.n_data), dtype=int)
        for r in range(self.n_data):
            rx, ry = divmod(r, self.m)
            for exp_x, exp_y in poly.terms:
                new_rx = (rx + exp_x) % self.l
                new_ry = (ry + exp_y) % self.m

                c = new_rx * self.m + new_ry
                matrix[r, c] ^= 1

        return matrix

    def _check_commutation(self, hx: np.ndarray, hz: np.ndarray):

        return np.all(hx @ hz.T == 0)

    @classmethod
    def generate_code(cls, distance: int):
        SUPPORTED_BB = {
            4: {
                "l": 3,
                "m": 3,
                "poly_a": "x^1 + y^0 + y^2",
                "poly_b": "y^1 + x^0 + x^2",
            },
            6: {
                "l": 6,
                "m": 6,
                "poly_a": "x^3 + y^1 + y^2",
                "poly_b": "y^3 + x^1 + x^2",
            },
            12: {
                "l": 12,
                "m": 6,
                "poly_a": "x^3 + y^1 + y^2",
                "poly_b": "y^3 + x^1 + x^2",
            },
            18: {
                "l": 12,
                "m": 12,
                "poly_a": "x^3 + y^2 + y^7",
                "poly_b": "y^3 + x^1 + x^2",
            },
        }

        if distance in SUPPORTED_BB:
            return cls(
                l=SUPPORTED_BB[distance]["l"],
                m=SUPPORTED_BB[distance]["m"],
                poly_a=Polynomial(SUPPORTED_BB[distance]["poly_a"]),
                poly_b=Polynomial(SUPPORTED_BB[distance]["poly_b"]),
                d=distance,
            )
        else:
            raise NotImplementedError(f"Distance: {distance} is not supported.")

class QubitType(enum.Enum):
    L = "L"
    R = "R"
    X = "X"
    Z = "Z"

class Basis(enum.Enum):
    X = "X"
    Z = "Z"

class BBAncillaQubit:
    def __init__(
        self,
        coord: tuple[int, int],
        id: int,
        l: int,
        m: int,
        poly_a_terms: list[tuple[int, int]],
        poly_b_terms: list[tuple[int, int]],
        basis: Basis,
    ):
        self.stabilizer_type = basis
        self.poly_a_terms = poly_a_terms
        self.poly_b_terms = poly_b_terms
        self.l = l
        self.m = m

        self.coord = coord
        self.id = id
        self.data_qubits = {}

    def add_associated_data_qubits(self, data_qubit):

        for i in range(len(data_qubit)):
            self.data_qubits[i] = data_qubit[i]


class DataQubit:
    def __init__(self, coord: tuple[int, int], id: int, qubit_type: QubitType):
        self.coord = coord
        self.id = id
        self.qubit_type = qubit_type



class MeasurementBookKeeper:

    def __init__(self, qubits: list = None):
        self.measurement_book_keeping = {}
        self.qubits = qubits if qubits is not None else []

        for q_id in self.qubits:
            self.measurement_book_keeping[q_id] = []

        self.measurement_count = 0

    def register_qubits(self, qubits: list):
        """Adds qubits to be tracked without resetting measurement_count, so a
        single book keeper can be shared across modules composed into the
        same circuit (their record indices must all be relative to one
        global measurement count)."""
        for q_id in qubits:
            if q_id not in self.measurement_book_keeping:
                self.measurement_book_keeping[q_id] = []
                self.qubits.append(q_id)

    def record_measurement(self, q_id: int):
        if q_id not in self.measurement_book_keeping:
            raise ValueError(f"Qubit ID {q_id} not found in measurement book keeping.")
        
        self.measurement_book_keeping[q_id].append(self.measurement_count)
        self.measurement_count += 1

    def get_last_measurement(self, q_id: int) -> int:
        if q_id not in self.measurement_book_keeping:
            raise ValueError(f"Qubit ID {q_id} not found in measurement book keeping.")
        if len(self.measurement_book_keeping[q_id]) == 0:
            raise ValueError(f"No measurements recorded for qubit ID {q_id}.")
        return self.measurement_book_keeping[q_id][-1]

    def get_last_two_measurements(self, q_id: int) -> tuple[int, int]:
        if q_id not in self.measurement_book_keeping:
            raise ValueError(f"Qubit ID {q_id} not found in measurement book keeping.")
        if len(self.measurement_book_keeping[q_id]) < 2:
            raise ValueError(f"Less than two measurements recorded for qubit ID {q_id}.")
        return self.measurement_book_keeping[q_id][-2], self.measurement_book_keeping[q_id][-1]


    def get_detection_event(self, q_id: int) -> int | tuple[int, int]:
        if q_id not in self.measurement_book_keeping:
            raise ValueError(f"Qubit ID {q_id} not found in measurement book keeping.")

        if len(self.measurement_book_keeping[q_id]) == 1:
            return self.get_last_measurement(q_id) - self.measurement_count
        elif len(self.measurement_book_keeping[q_id]) >= 2:
            return tuple(m - self.measurement_count for m in self.get_last_two_measurements(q_id))
        
        else:
            raise ValueError(f"No measurements recorded for qubit ID {q_id}.")
    
class BBCodeModule:

    def __init__(self,
                 code: BivariateBicycleCode,
                 basis: Basis,
                 *,
                 start_idx=0,
                 measurement_book_keeper: "MeasurementBookKeeper" = None,
        ):
        self.code = code
        self.basis = basis
        self.start_idx = start_idx

        self.x_check_offset = start_idx
        self.l_data_offset = start_idx + self.code.l * self.code.m
        self.r_data_offset = start_idx + 2 * self.code.l * self.code.m
        self.z_data_offset = start_idx + 3 * self.code.l * self.code.m

        self.data_qubits = {}
        self.l_data_qubits = []
        self.r_data_qubits = []
        self.ancilla_qubits = {
            Basis.X : [],
            Basis.Z : []
        }
        self.x_check_qubits = []
        self.z_check_qubits = []
        self._init_data_qubits()
        self._init_ancillas(Basis.X)
        self._init_ancillas(Basis.Z)

        all_ids = [q.id for q in self.l_data_qubits + self.r_data_qubits + self.x_check_qubits + self.z_check_qubits]
        if measurement_book_keeper is not None:
            self.measurement_book_keeper = measurement_book_keeper
            self.measurement_book_keeper.register_qubits(all_ids)
        else:
            self.measurement_book_keeper = MeasurementBookKeeper(qubits=all_ids)

        self.ancilla_system = {}
    
    def qubit_offset_map(self, qubit_type: QubitType) -> int:
        if qubit_type == QubitType.L:
            return self.l_data_offset
        elif qubit_type == QubitType.R:
            return self.r_data_offset
        elif qubit_type == QubitType.Z:
            return self.z_data_offset
        elif qubit_type == QubitType.X:
            return self.x_check_offset
        else:
            raise ValueError(f"Unknown qubit type: {qubit_type}")


    

    
    def _init_data_qubits(self):

        for i in range(self.code.l * self.code.m):

            l_idx = i + self.qubit_offset_map(QubitType.L)
            r_idx = i + self.qubit_offset_map(QubitType.R)

            l_qubit = DataQubit(
                coord=(-1, *self.get_qubit_torus(i, QubitType.L)),
                id=l_idx,
                qubit_type=QubitType.L
            )
            r_qubit = DataQubit(
                coord=(0, *self.get_qubit_torus(i, QubitType.R)),
                id=r_idx,
                qubit_type=QubitType.R
            )

            self.data_qubits[l_idx] = l_qubit
            self.data_qubits[r_idx] = r_qubit

            self.l_data_qubits.append(l_qubit)
            self.r_data_qubits.append(r_qubit)

    def get_qubit_from_torus(
            self, data_qubit_type: QubitType, torus_coords: tuple[int, int]
        ) -> DataQubit:
        """
        Get the data qubit corresponding to the given torus coordinates and qubit type.
        Torus coordinates (i, j) correspond to the monomial exponents of x^iy^j.
        
        Args:
            data_qubit_type (QubitType): The type of the data qubit (L, R, X, Z).
            torus_coords (tuple[int, int]): The coordinates (i, j) on the torus.

        Returns:
            DataQubit: The data qubit at the specified coordinates and type.
        """
        torus_coords = self._resolve_monomial_exponents(torus_coords, (0, 0))
        idx = self.get_qubit_idx(data_qubit_type, torus_coords)
        return self.data_qubits[idx] 
    

    def get_qubit_idx(self, data_qubit_type: QubitType, torus_coords: tuple[int, int]) -> int:
        i, j = torus_coords
        return i * self.code.m + j + self.qubit_offset_map(data_qubit_type)
    
    def get_qubit_torus(self, idx: int, data_qubit_type: QubitType) -> tuple[int, int]:
        """
         Get the torus coordinates (i, j) for the data qubit with the given index.
         Returns a tuple (i, j) representing the coordinates on the torus.
        """
        idx -= self.qubit_offset_map(data_qubit_type)

        i = idx // self.code.m
        j = idx % self.code.m
        return (i, j)

    


    def _resolve_monomial_exponents(self, 
                                    monomial_exponent: tuple[int, int], 
                                    polynomial_term: tuple[int, int],
    ):
        
        l, m = self.code.l, self.code.m
        i, j = monomial_exponent
        ii, jj = polynomial_term
        return ((i + ii) % l, (j + jj) % m)

    
    def _init_ancillas(self, basis: Basis):
        coord_second_part_offset = 0 if basis == Basis.X else self.code.n // 2
        id_offset = 0 if basis == Basis.X else 3 * self.code.n // 2
        data_qubit_offset = self.code.n // 2

        poly_a = self.code.poly_a
        poly_b = self.code.poly_b

        ancillas = []
        for i in range(self.code.l):
            for j in range(self.code.m):

                if basis == Basis.X:
                    # 1. Forward shifts for X
                    terms_a = [
                        self._resolve_monomial_exponents(
                            monomial_exponent=(i, j),
                            polynomial_term=t,
                        )
                        for t in poly_a.terms
                    ]
                    terms_b = [
                        self._resolve_monomial_exponents(
                            monomial_exponent=(i, j),
                            polynomial_term=t,
                        )
                        for t in poly_b.terms
                    ]

                    # 2. A -> Left Data (offset 0), B -> Right Data (offset n_data)
                    A_targets = [
                        self.get_qubit_from_torus(QubitType.L, t)
                        for t in terms_a
                    ]
                    B_targets = [
                        self.get_qubit_from_torus(QubitType.R, t)
                        for t in terms_b
                    ]

                    # 3. X schedule: A2, B2, B1, B3, A1, A3
                    ordered = [
                        A_targets[1],
                        B_targets[1],
                        B_targets[0],
                        B_targets[2],
                        A_targets[0],
                        A_targets[2],
                    ]

                else:  # Basis.Z
                    # 1. Transposed (reverse) shifts for Z
                    terms_a = [
                        self._resolve_monomial_exponents(
                            monomial_exponent=(i, j),
                            polynomial_term=(-t[0], -t[1]),
                        )
                        for t in poly_a.terms
                    ]
                    terms_b = [
                        self._resolve_monomial_exponents(
                            monomial_exponent=(i, j),
                            polynomial_term=(-t[0], -t[1]),
                        )
                        for t in poly_b.terms
                    ]

                    # 2. A^T -> Right Data (offset n_data), B^T -> Left Data (offset 0)
                    A_targets = [
                        self.get_qubit_from_torus(QubitType.R, t)
                        for t in terms_a
                    ]
                    B_targets = [
                        self.get_qubit_from_torus(QubitType.L, t)
                        for t in terms_b
                    ]

                    # 3. Z schedule: A1^T, A3^T, B1^T, B2^T, B3^T, A2^T
                    ordered = [
                        A_targets[0],
                        A_targets[2],
                        B_targets[0],
                        B_targets[1],
                        B_targets[2],
                        A_targets[1],
                    ]

                anc = BBAncillaQubit(
                    coord=(1, i, j),
                    id=id_offset + i * self.code.m + j,
                    basis=basis,
                    l=self.code.l,
                    m=self.code.m,
                    poly_a_terms=terms_a,
                    poly_b_terms=terms_b,
                )

                anc.add_associated_data_qubits(ordered)
                ancillas.append(anc)
        if basis == Basis.X:
            self.x_check_indices = [anc.id for anc in ancillas]
            self.x_check_qubits = ancillas
        else:
            self.z_check_indices = [anc.id for anc in ancillas]
            self.z_check_qubits = ancillas
        self.ancilla_qubits[basis] = ancillas


    def initial_round(self):

        circ = stim.Circuit()

        def _qubit_coord_text(qubits, start_idx):
            text = ""
            for q in qubits:
                text += f"QUBIT_COORDS(0, {q.id}) {q.id}\n"
    
            return text

        idx_counter = 0
        circ.append_from_stim_program_text(
            _qubit_coord_text(self.x_check_qubits, start_idx=idx_counter)
        )
        idx_counter += len(self.x_check_qubits)

        circ.append_from_stim_program_text(
            _qubit_coord_text(self.l_data_qubits, start_idx=idx_counter)
        )
        idx_counter += len(self.l_data_qubits)

        circ.append_from_stim_program_text(
            _qubit_coord_text(self.r_data_qubits, start_idx=idx_counter)
        )
        idx_counter += len(self.r_data_qubits)

        circ.append_from_stim_program_text(
            _qubit_coord_text(self.z_check_qubits, start_idx=idx_counter)
        )
        idx_counter += len(self.z_check_qubits)


        circ.append("R", [q.id for q in self.x_check_qubits + self.l_data_qubits + self.r_data_qubits + self.z_check_qubits])
        if self.basis == Basis.X:
            circ.append("H", [q.id for q in self.l_data_qubits + self.r_data_qubits])
        circ.append("TICK")
        return circ

    def _yield_a_stage_syndrome_ex(
        self, stage: int, x_checks, z_checks
    ):

        def get_interaction_pair(anc: BBAncillaQubit, stage:int):
            
            if anc.stabilizer_type == Basis.X:
                return anc.id, anc.data_qubits[stage].id
            elif anc.stabilizer_type == Basis.Z:
                return anc.data_qubits[stage].id, anc.id


        z_stage = stage
        x_stage = stage - 1

        ops = {
            Basis.X: {
                "control": [],
                "target": [],
            },
            Basis.Z: {
                "control": [],
                "target": [],
            },
        }

        if z_stage <= 5:
            for anc in z_checks:
                ctrl, targ = get_interaction_pair(anc, z_stage)
                if ctrl is not None and targ is not None:
                    ops[Basis.Z]["control"].append(ctrl)
                    ops[Basis.Z]["target"].append(targ)

        if x_stage >= 0:
            for anc in x_checks:
                ctrl, targ = get_interaction_pair(anc, x_stage)
                if ctrl is not None and targ is not None:
                    ops[Basis.X]["control"].append(ctrl)
                    ops[Basis.X]["target"].append(targ)

        return (
            ops[Basis.X]["control"],
            ops[Basis.X]["target"],
            ops[Basis.Z]["control"],
            ops[Basis.Z]["target"],
        )

    def syndrome_extraction(self, circ: stim.Circuit):
        circ.append("H", [q.id for q in self.x_check_qubits])
        
        for step in range(7):
            x_ctrls, x_targs, z_ctrls, z_targs = self._yield_a_stage_syndrome_ex(
                step, self.x_check_qubits, self.z_check_qubits
            )

            pairs = []
            for ct in zip(x_ctrls, x_targs):
                pairs.append(ct[0])
                pairs.append(ct[1])
            for ct in zip(z_ctrls, z_targs):
                pairs.append(ct[0])
                pairs.append(ct[1])

            circ.append("CX", pairs)
            circ.append("TICK")

        circ.append("H", [q.id for q in self.x_check_qubits])

        return circ

    def measurement(self, circ, qubits):

        measurement_q_ids = [q.id for q in qubits]
        circ.append("MR", measurement_q_ids)
        
        for q_id in measurement_q_ids:
            self.measurement_book_keeper.record_measurement(q_id) 

        return circ
    
    def checks_detection_events(self, checks):
        instructions = []
        for q in checks:
            event = self.measurement_book_keeper.get_detection_event(q.id)
            
            if isinstance(event, int):
                instr = stim.CircuitInstruction("DETECTOR", [stim.target_rec(event)], (0, 0, q.id))
                instructions.append(instr)
            elif isinstance(event, tuple) or isinstance(event, list):
                instr = stim.CircuitInstruction("DETECTOR", [stim.target_rec(event[0]), stim.target_rec(event[1])], (0, 0, q.id))
                instructions.append(instr)

        return instructions

    def qec_round(self, circ: stim.Circuit, detector_basis: list[Basis]):
        circ = self.syndrome_extraction(circ)
        circ = self.measurement(circ, self.x_check_qubits + self.z_check_qubits)

        checks = []
        if Basis.X in detector_basis:
            checks += self.x_check_qubits
        if Basis.Z in detector_basis:
            checks += self.z_check_qubits

        instructions = self.checks_detection_events(checks)
        for instr in instructions:
            circ.append(instr)

        circ.append("SHIFT_COORDS", [], [1, 0, 0]) 
        return circ

    def measure_all_data_qubits(self, circ):
        if self.basis == Basis.X:
            circ.append("H", [q.id for q in self.l_data_qubits + self.r_data_qubits])

        return self.measurement(circ, self.l_data_qubits + self.r_data_qubits)

    def _logical_observable(self):

        # logical_matrix = self.bb.lz if self.basis == Basis.Z else self.bb.lx
        if self.basis.value == Basis.Z.value:
            logical_matrix = self.code.lz
        elif self.basis.value == Basis.X.value:
            logical_matrix = self.code.lx
        else:
            raise ValueError(f"Basis {self.basis} not supported")

        n = self.code.n

        for obs_indx, logical_op in enumerate(logical_matrix):
            involved_qubits = np.nonzero(logical_op)[1]
            # print(involved_qubits)
            yield involved_qubits

    def observable(self, circ):

        for idx, ops in enumerate(self._logical_observable()):
            targets = []
            for qubit_index in ops:
                targets.append(
                    self.measurement_book_keeper.get_detection_event(self.data_qubits[qubit_index + self.code.n // 2].id))
            circ.append("OBSERVABLE_INCLUDE", [stim.target_rec(tar) for tar in targets], idx)

        return circ
    
    def finalize(self, circ: stim.Circuit):

        checks = self.x_check_qubits if self.basis == Basis.X else self.z_check_qubits

        for i, anc in enumerate(checks):
            targets = []
            for direction, neighbor in anc.data_qubits.items():
                if neighbor is not None:
                    targets.append(
                        self.measurement_book_keeper.get_last_measurement(neighbor.id) - self.measurement_book_keeper.measurement_count
                    )
            targets.append(
                self.measurement_book_keeper.get_last_measurement(anc.id) - self.measurement_book_keeper.measurement_count
            )
            targets = [stim.target_rec(tar) for tar in targets]

            circ.append("DETECTOR", targets, (0, 0, anc.id))

        return circ
    
    

    def _logical_op_support(self, logical_matrix, logical_index: int) -> list[int]:
        """Maps a row of code.lx/code.lz to the physical qubit ids it acts
        on, honoring start_idx (unlike observable()'s "+ code.n // 2" trick,
        which only works when start_idx == 0)."""
        n_data = self.code.l * self.code.m
        involved = np.nonzero(logical_matrix[logical_index])[1]
        ids = []
        for idx in involved:
            if idx < n_data:
                ids.append(self.data_qubits[self.l_data_offset + idx].id)
            else:
                ids.append(self.data_qubits[self.r_data_offset + (idx - n_data)].id)
        return ids

    def get_logical_x_support(self, logical_index: int) -> list[int]:
        return self._logical_op_support(self.code.lx, logical_index)

    def get_logical_z_support(self, logical_index: int) -> list[int]:
        return self._logical_op_support(self.code.lz, logical_index)
    
    

    def get_qubit_support_X(self, a: Polynomial, b: Polynomial):
        """
        We denote X(a, b) as the Pauli X operator acting on the qubits 
        q(L, α), α in a and 
        q(R, β), β in b.

        Returns the set of qubits that support X(a, b),
        where a is the polynomial corresponding to the left data qubits 
        and b corresponds to the right data qubits.
        """

        qubits = []
        for term in a.terms:
            qubits.append(self.get_qubit_from_torus(QubitType.L, term))
        for term in b.terms:
            qubits.append(self.get_qubit_from_torus(QubitType.R, term))
        return qubits
    
    def get_involved_checks(self, qubits, check_matrix):
        """
        Given a set of data qubits, return the row indices of
        `check_matrix` (code.hx or code.hz) that touch at least one of
        them -- i.e. have a nonzero entry in the qubit's column.

        `qubits` may be DataQubit objects (as returned by
        get_qubit_support_X / get_logical_x_support / etc., whose `.id`
        is a module-level id offset by l_data_offset/r_data_offset) or
        plain ints already in check_matrix's raw column space
        (0 <= idx < 2*l*m, L block then R block, matching code.hx/hz).
        """
        n_data = self.code.l * self.code.m

        columns = []
        for q in qubits:
            qid = q.id if isinstance(q, DataQubit) else q
            if self.l_data_offset <= qid < self.l_data_offset + n_data:
                columns.append(qid - self.l_data_offset)
            elif self.r_data_offset <= qid < self.r_data_offset + n_data:
                columns.append(n_data + (qid - self.r_data_offset))
            elif 0 <= qid < 2 * n_data:
                columns.append(qid)
            else:
                raise ValueError(
                    f"Qubit id {qid} is not a data qubit (L/R) of this module."
                )

        if not columns:
            return np.array([], dtype=int)

        check_matrix = sparse.csr_matrix(check_matrix)
        restricted_columns = check_matrix[:, columns]
        return np.unique(restricted_columns.nonzero()[0])
    
    def ancilla_system_for_one_logical_operator(
        self,
        logical_matrix,
        logical_index: int,
        check_matrix,
    ):
        """
        Construct the Tanner subgraph G_lx for one logical operator.

        For a logical-X operator:
            logical_matrix = code.lx
            check_matrix = code.hz

        Returns:
            involved:
                Original data-qubit column indices in the logical support
                V_lx.

            check_indices:
                Check-row indices touching the logical support
                C_lx.

            M:
                Restricted incidence matrix
                check_matrix[C_lx, V_lx].
        """

        if not 0 <= logical_index < logical_matrix.shape[0]:
            raise IndexError(
                f"logical_index={logical_index} is outside "
                f"[0, {logical_matrix.shape[0]})"
            )

        if logical_matrix.shape[1] != check_matrix.shape[1]:
            raise ValueError(
                "logical_matrix and check_matrix must act on the same "
                f"number of data qubits, but got {logical_matrix.shape[1]} "
                f"and {check_matrix.shape[1]} columns."
            )

        # Convert to CSR to make row slicing efficient.
        logical_matrix = sparse.csr_matrix(logical_matrix)
        check_matrix = sparse.csr_matrix(check_matrix)

        # V_lx: columns where the selected logical operator is nonzero.
        logical_row = logical_matrix.getrow(logical_index)
        involved = logical_row.indices.copy()

        if involved.size == 0:
            raise ValueError(
                f"Logical operator {logical_index} has empty support."
            )

        # Restrict the check matrix to the logical support.
        #
        # Its nonzero rows are exactly the checks touching at least one
        # qubit in V_lx.
        restricted_columns = check_matrix[:, involved]

        # C_lx: rows containing at least one nonzero entry.
        check_indices = np.unique(restricted_columns.nonzero()[0])

        # M = H_check[C_lx, V_lx].
        M = check_matrix[check_indices, :][:, involved].tocsr()

        print("Logical support V_lx:", involved.tolist())
        print("Touching checks C_lx:", check_indices.tolist())
        print("Restricted incidence matrix M:")
        print(M.toarray())

        return involved, check_indices, M
    
    def _data_qubit_to_code_column(self, qubit: DataQubit) -> int:
        """
        Convert a physical DataQubit into its column index in Hx/Hz.

        Hx and Hz use the column ordering:
            [L data | R data].
        """
        n_data = self.code.n_data

        if qubit.qubit_type == QubitType.L:
            column = qubit.id - self.l_data_offset

            if not 0 <= column < n_data:
                raise ValueError(
                    f"Invalid L-data qubit ID {qubit.id}: "
                    f"computed local column {column}."
                )

            return column

        if qubit.qubit_type == QubitType.R:
            local_column = qubit.id - self.r_data_offset

            if not 0 <= local_column < n_data:
                raise ValueError(
                    f"Invalid R-data qubit ID {qubit.id}: "
                    f"computed local column {local_column}."
                )

            return n_data + local_column

        raise ValueError(
            f"Qubit {qubit.id} has type {qubit.qubit_type}, "
            "but only L and R data qubits can be matrix columns."
        )
    
    def construct_incidence_matrix(
        self,
        data_vertices: list[DataQubit],
        check_matrix,
    ):
        """
        Construct the restricted Tanner incidence matrix

            M = H_check[C, V],

        where V is the logical-operator support and C contains all checks
        touching V.

        Returns:
            data_columns:
                Column indices V in the original parity-check matrix.

            check_rows:
                Row indices C in the original parity-check matrix.

            M:
                Restricted sparse incidence matrix.

        Raises:
            ValueError if M @ 1 != 0 over GF(2).
        """
        if not data_vertices:
            raise ValueError("The logical operator has empty physical support.")

        H = sparse.csr_matrix(
            np.asarray(check_matrix, dtype=np.uint8) & 1
        )

        # Convert physical qubit objects into Hx/Hz column indices.
        data_columns = np.array(
            [
                self._data_qubit_to_code_column(qubit)
                for qubit in data_vertices
            ],
            dtype=np.int64,
        )

        # Duplicate vertices would incorrectly cancel or duplicate columns.
        if len(np.unique(data_columns)) != len(data_columns):
            unique_columns, counts = np.unique(
                data_columns,
                return_counts=True,
            )
            duplicates = unique_columns[counts > 1]

            raise ValueError(
                "The logical support contains duplicate data vertices. "
                f"Duplicate matrix columns: {duplicates.tolist()}"
            )

        # H[:, V].
        restricted_columns = H[:, data_columns]

        # C consists of all rows that touch at least one vertex in V.
        check_rows = np.unique(
            restricted_columns.nonzero()[0]
        ).astype(np.int64)

        # M = H[C, V].
        M = H[check_rows, :][:, data_columns].tocsr()

        # Normalize stored entries over GF(2).
        M.data %= 2
        M.eliminate_zeros()

        # Verify M * 1 = 0 over GF(2).
        ones = np.ones(M.shape[1], dtype=np.uint8)
        row_parities = np.asarray(M @ ones).reshape(-1) % 2

        bad_local_rows = np.flatnonzero(row_parities)

        if bad_local_rows.size:
            bad_original_rows = check_rows[bad_local_rows]

            details = []
            for local_row, original_row in zip(
                bad_local_rows[:10],
                bad_original_rows[:10],
            ):
                touching_local_columns = M.getrow(local_row).indices
                touching_code_columns = data_columns[touching_local_columns]

                details.append(
                    {
                        "check_row": int(original_row),
                        "overlap_weight": int(
                            M.getrow(local_row).nnz
                        ),
                        "logical_support_columns": (
                            touching_code_columns.tolist()
                        ),
                    }
                )

            raise ValueError(
                "The restricted incidence matrix does not satisfy "
                "M @ 1 = 0 over GF(2). This means at least one "
                "opposite-type check overlaps the proposed logical "
                "operator an odd number of times. "
                f"Bad original check rows: "
                f"{bad_original_rows.tolist()}. "
                f"First details: {details}"
            )

        print(
            "Verified incidence matrix: "
            f"M has shape {M.shape}, nnz={M.nnz}, "
            "and M @ 1 = 0 over GF(2)."
        )

        return data_columns, check_rows, M
        
    def reset_ancilla_system(self):
        self.ancilla_system.clear()

        self.primal_qubits_map = {}
        self.dual_qubits_map = {}

        # These lists contain only newly allocated deformation-data qubits.
        # p_1(v) is not included because it is an original BB data qubit.
        self.primal_qubit_list = []
        self.dual_qubit_list = []
    def _next_free_qubit_id(self) -> int:
        """
        Return the first unused physical qubit ID.

        This remains correct when start_idx is nonzero.
        """
        existing_ids = [
            q.id
            for q in (
                self.x_check_qubits
                + self.l_data_qubits
                + self.r_data_qubits
                + self.z_check_qubits
            )
        ]

        if not existing_ids:
            return self.start_idx

        return max(existing_ids) + 1

    def build_primal_data_qubits(
        self,
        first_layer_qubits: list[DataQubit],
        data_columns: np.ndarray,
        next_free_id: int,
    ) -> int:
        """
        Allocate p_j(v).

        Args:
            first_layer_qubits:
                Original BB data qubits supporting X(poly_1, poly_2).

            data_columns:
                Corresponding columns v in the BB parity-check matrices.

            next_free_id:
                First available physical qubit ID.

        Returns:
            The next unused physical qubit ID after allocation.

        Map convention:
            primal_qubits_map[j][v] = physical qubit ID of p_j(v).

        Layer 1:
            p_1(v) is the original BB data qubit.

        Layers 2,...,r:
            p_j(v) is a newly allocated deformation-data qubit.
        """
        data_columns = np.asarray(data_columns, dtype=np.int64).reshape(-1)

        if len(first_layer_qubits) != len(data_columns):
            raise ValueError(
                "first_layer_qubits and data_columns have different lengths: "
                f"{len(first_layer_qubits)} and {len(data_columns)}."
            )

        if len(np.unique(data_columns)) != len(data_columns):
            raise ValueError("data_columns contains duplicate Tanner vertices.")

        self.primal_qubits_map = {}
        self.primal_qubit_list = []

        # G[1]: original BB data qubits.
        self.primal_qubits_map[1] = {
            int(v): int(qubit.id)
            for v, qubit in zip(data_columns, first_layer_qubits)
        }

        # G[2], ..., G[r]: new physical deformation-data qubits.
        for layer in range(2, self.code.d + 1):
            self.primal_qubits_map[layer] = {}

            for v in data_columns:
                v = int(v)

                self.primal_qubits_map[layer][v] = next_free_id
                self.primal_qubit_list.append(next_free_id)
                next_free_id += 1

        return next_free_id
        

    def build_dual_data_qubits(
        self,
        check_rows: np.ndarray,
        next_free_id: int,
    ) -> int:
        """
        Allocate d_j(c) for j=1,...,r.

        Args:
            check_rows:
                Original opposite-type check rows c belonging to C.

            next_free_id:
                First available physical qubit ID.

        Returns:
            The next unused physical qubit ID after allocation.

        Map convention:
            dual_qubits_map[j][c] = physical qubit ID of d_j(c).

        Important:
            Every d_j(c), including d_1(c), is a newly allocated physical
            deformation-data qubit.
        """
        check_rows = np.asarray(check_rows, dtype=np.int64).reshape(-1)

        if len(np.unique(check_rows)) != len(check_rows):
            raise ValueError("check_rows contains duplicate Tanner vertices.")

        self.dual_qubits_map = {}
        self.dual_qubit_list = []

        for layer in range(1, self.code.d + 1):
            self.dual_qubits_map[layer] = {}

            for c in check_rows:
                c = int(c)

                self.dual_qubits_map[layer][c] = next_free_id
                self.dual_qubit_list.append(next_free_id)
                next_free_id += 1

        return next_free_id
    
    def build_ancilla_system(
        self,
        poly_1: Polynomial,
        poly_2: Polynomial,
    ):
        """
        Construct the primal/dual ancilla data for X(poly_1, poly_2).

        The polynomial condition is

            [B | A] [g] = Bg + Ah = 0 over GF(2),
                    [h]

        where:
            poly_1 = g
            poly_2 = h

        For X(f, 0), pass:
            poly_1 = f
            poly_2 = zero polynomial

        The constructed data maps are:

            primal_qubits_map[j][v] = physical ID of p_j(v)
            dual_qubits_map[j][c]   = physical ID of d_j(c)
        """
        self.reset_ancilla_system()

        B = np.asarray(self.code.B, dtype=np.uint8) & 1
        A = np.asarray(self.code.A, dtype=np.uint8) & 1

        g = np.asarray(
            self.code._poly_to_matrix(poly_1),
            dtype=np.uint8,
        ) & 1

        h = np.asarray(
            self.code._poly_to_matrix(poly_2),
            dtype=np.uint8,
        ) & 1

        lm = B.shape[0]
        expected_shape = (lm, lm)

        for name, matrix in {
            "A": A,
            "B": B,
            "g": g,
            "h": h,
        }.items():
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {matrix.shape}; "
                    f"expected {expected_shape}."
                )

        # Verify [B | A][g; h] = Bg + Ah = 0 over GF(2).
        BA = np.hstack([B, A])
        gh = np.vstack([g, h])

        residual = (BA @ gh) % 2
        direct_residual = (B @ g + A @ h) % 2

        if not np.array_equal(residual, direct_residual):
            raise RuntimeError(
                "Internal inconsistency when checking Bg + Ah."
            )

        if np.any(residual):
            nonzero_entries = np.argwhere(residual)

            raise ValueError(
                "The supplied polynomials do not satisfy "
                "[B | A][g; h] = Bg + Ah = 0 over GF(2). "
                f"The residual has weight {np.count_nonzero(residual)}. "
                f"First nonzero entries: "
                f"{nonzero_entries[:10].tolist()}"
            )

        # V: physical support of X(poly_1, poly_2).
        V = self.get_qubit_support_X(poly_1, poly_2)

        if not V:
            raise ValueError("The proposed logical operator has empty support.")

        physical_support_ids = [q.id for q in V]
        if len(set(physical_support_ids)) != len(physical_support_ids):
            raise ValueError(
                "The physical logical support contains duplicate qubits."
            )

        # For an X-type logical operator, use neighboring Z checks.
        #
        # V_columns:
        #     Original H_Z columns representing Tanner data vertices v.
        #
        # C_rows:
        #     Original H_Z rows representing Tanner check vertices c.
        #
        # M:
        #     Restricted incidence matrix H_Z[C, V].
        V_columns, C_rows, M = self.construct_incidence_matrix(
            data_vertices=V,
            check_matrix=self.code.hz,
        )

        if M.shape != (len(C_rows), len(V_columns)):
            raise RuntimeError(
                f"Incidence matrix has shape {M.shape}; expected "
                f"({len(C_rows)}, {len(V_columns)})."
            )

        # Verify M * 1 = 0 over GF(2).
        ones = np.ones(M.shape[1], dtype=np.uint8)
        row_parities = np.asarray(M @ ones).reshape(-1) % 2

        bad_local_rows = np.flatnonzero(row_parities)
        if bad_local_rows.size:
            bad_original_rows = C_rows[bad_local_rows]

            raise ValueError(
                "The restricted incidence matrix does not satisfy "
                "M @ 1 = 0 over GF(2). "
                f"Odd-overlap original check rows: "
                f"{bad_original_rows.tolist()}."
            )

        # Allocate all spatial layers.
        next_free_id = self._next_free_qubit_id()

        next_free_id = self.build_primal_data_qubits(
            first_layer_qubits=V,
            data_columns=V_columns,
            next_free_id=next_free_id,
        )

        next_free_id = self.build_dual_data_qubits(
            check_rows=C_rows,
            next_free_id=next_free_id,
        )

        new_data_ids = (
            self.primal_qubit_list
            + self.dual_qubit_list
        )

        # Check for allocation collisions.
        if len(new_data_ids) != len(set(new_data_ids)):
            raise RuntimeError(
                "Duplicate physical IDs were allocated in the ancilla system."
            )

        original_ids = {
            q.id
            for q in (
                self.x_check_qubits
                + self.l_data_qubits
                + self.r_data_qubits
                + self.z_check_qubits
            )
        }

        collisions = original_ids.intersection(new_data_ids)
        if collisions:
            raise RuntimeError(
                "Ancilla-system data IDs collide with original code IDs: "
                f"{sorted(collisions)}"
            )

        # Register added qubits if their measurements will be recorded.
        self.measurement_book_keeper.register_qubits(new_data_ids)

        # Store the abstract graph and physical-layer allocation.
        self.ancilla_system.update({
            "basis": Basis.X,
            "logical_polynomials": (poly_1, poly_2),

            # Abstract first primal graph G[1].
            "G1": {
                "data_columns": V_columns.copy(),
                "data_qubits": list(V),
                "check_rows": C_rows.copy(),
                "incidence_matrix": M.copy(),
            },

            # Abstract first dual graph G^T[1].
            #
            # Original check vertices c become dual data vertices.
            # Original data vertices v become dual check vertices.
            "GT1": {
                "data_vertices": C_rows.copy(),
                "check_vertices": V_columns.copy(),
                "incidence_matrix": M.T.tocsr(),
            },

            # Physical realizations of p_j(v) and d_j(c).
            "primal_data": self.primal_qubits_map,
            "dual_data": self.dual_qubits_map,

            "new_primal_data_ids": list(self.primal_qubit_list),
            "new_dual_data_ids": list(self.dual_qubit_list),
            "new_data_qubit_ids": list(new_data_ids),
            "next_free_qubit_id": next_free_id,
        })

        expected_new_primal = (
            (self.code.d - 1) * len(V_columns)
        )
        expected_new_dual = (
            self.code.d * len(C_rows)
        )

        if len(self.primal_qubit_list) != expected_new_primal:
            raise RuntimeError(
                "Incorrect number of primal data qubits: "
                f"got {len(self.primal_qubit_list)}, "
                f"expected {expected_new_primal}."
            )

        if len(self.dual_qubit_list) != expected_new_dual:
            raise RuntimeError(
                "Incorrect number of dual data qubits: "
                f"got {len(self.dual_qubit_list)}, "
                f"expected {expected_new_dual}."
            )

        print(
            "Built ancilla system:\n"
            f"  |V| = {len(V_columns)}\n"
            f"  |C| = {len(C_rows)}\n"
            f"  M shape = {M.shape}\n"
            f"  spatial depth r = {self.code.d}\n"
            f"  new primal data = {len(self.primal_qubit_list)}\n"
            f"  new dual data = {len(self.dual_qubit_list)}\n"
            f"  total new data = {len(new_data_ids)}"
        )

        return self.ancilla_system

    def _to_binary_csr(self, matrix) -> sparse.csr_matrix:
        """
        Convert a dense, Galois, or SciPy sparse matrix into a CSR matrix
        whose entries are reduced modulo 2.
        """
        if sparse.issparse(matrix):
            result = matrix.astype(np.uint8).tocsr(copy=True)
        else:
            result = sparse.csr_matrix(
                np.asarray(matrix, dtype=np.uint8)
            )

        result.data %= 2
        result.eliminate_zeros()
        return result


    def _deformed_data_column_layout(
        self,
        ancilla_system=None,
    ):
        """
        Construct a compact parity-check-matrix column ordering.

        Columns:
            0, ..., n_original - 1:
                original BB data columns [L | R]

            n_original, ...:
                newly allocated deformation-data qubits

        Returns:
            number_of_columns
            physical_id_to_column
            column_to_physical_id
        """
        ancilla_system = (
            self.ancilla_system
            if ancilla_system is None
            else ancilla_system
        )

        new_data_ids = [
            int(qid)
            for qid in ancilla_system["new_data_qubit_ids"]
        ]

        if len(new_data_ids) != len(set(new_data_ids)):
            raise ValueError(
                "ancilla_system contains duplicate deformation-data IDs."
            )

        original_physical_ids = (
            [q.id for q in self.l_data_qubits]
            + [q.id for q in self.r_data_qubits]
        )

        column_to_physical_id = (
            original_physical_ids
            + new_data_ids
        )

        if len(column_to_physical_id) != len(
            set(column_to_physical_id)
        ):
            raise ValueError(
                "Original and deformation-data physical IDs overlap."
            )

        physical_id_to_column = {
            physical_id: column
            for column, physical_id in enumerate(
                column_to_physical_id
            )
        }

        number_of_columns = len(column_to_physical_id)

        # Store this because it will later be needed to turn matrix
        # supports into Stim physical-qubit targets.
        ancilla_system["data_column_to_physical_id"] = (
            column_to_physical_id
        )
        ancilla_system["physical_id_to_data_column"] = (
            physical_id_to_column
        )

        return (
            number_of_columns,
            physical_id_to_column,
            column_to_physical_id,
        )
    
    def new_x_checks(
        self,
        ancilla_system=None,
    ) -> sparse.csr_matrix:
        """
        Construct the complete X-check matrix Hx_deform for an
        X-type logical deformation.

        The returned matrix contains:

            1. All original BB X checks, padded with zero columns.
            2. The new checks A_{j,v} for j=1,...,r and v in V.

        A_{j,v} acts on:
            p_j(v),
            p_{j+1}(v) when j < r,
            d_j(c) for every c satisfying M[c,v] = 1.
        """
        ancilla_system = (
            self.ancilla_system
            if ancilla_system is None
            else ancilla_system
        )

        if not ancilla_system:
            raise ValueError(
                "No ancilla system has been constructed."
            )

        if ancilla_system.get("basis") != Basis.X:
            raise NotImplementedError(
                "This implementation currently constructs the "
                "X-type logical deformation."
            )

        G1 = ancilla_system["G1"]

        V = np.asarray(
            G1["data_columns"],
            dtype=np.int64,
        ).reshape(-1)

        C = np.asarray(
            G1["check_rows"],
            dtype=np.int64,
        ).reshape(-1)

        M = self._to_binary_csr(
            G1["incidence_matrix"]
        )

        if M.shape != (len(C), len(V)):
            raise ValueError(
                f"M has shape {M.shape}; expected "
                f"({len(C)}, {len(V)})."
            )

        primal = ancilla_system["primal_data"]
        dual = ancilla_system["dual_data"]

        primal_layers = sorted(int(j) for j in primal)
        dual_layers = sorted(int(j) for j in dual)

        if not primal_layers or not dual_layers:
            raise ValueError(
                "Missing primal or dual layer allocations."
            )

        r = max(primal_layers)

        expected_layers = list(range(1, r + 1))

        if primal_layers != expected_layers:
            raise ValueError(
                f"Primal layers are {primal_layers}; "
                f"expected {expected_layers}."
            )

        if dual_layers != expected_layers:
            raise ValueError(
                f"Dual layers are {dual_layers}; "
                f"expected {expected_layers}."
            )

        (
            total_columns,
            physical_to_column,
            _,
        ) = self._deformed_data_column_layout(
            ancilla_system
        )

        original_hx = self._to_binary_csr(
            self.code.hx
        )

        original_columns = original_hx.shape[1]

        if original_columns != self.code.n:
            raise ValueError(
                f"Original Hx has {original_columns} columns, "
                f"but code.n={self.code.n}."
            )

        padding = sparse.csr_matrix(
            (
                original_hx.shape[0],
                total_columns - original_columns,
            ),
            dtype=np.uint8,
        )

        padded_original_hx = sparse.hstack(
            [original_hx, padding],
            format="csr",
        )

        # Use CSC because we repeatedly query M[:, v].
        M_by_column = M.tocsc()

        row_indices = []
        column_indices = []
        row_metadata = []

        new_row = 0

        for layer in range(1, r + 1):
            for v_local, v in enumerate(V):
                v = int(v)
                support = set()

                # p_j(v)
                if layer == 1:
                    # p_1(v) is the original BB data column v.
                    support.add(v)
                else:
                    p_physical_id = int(
                        primal[layer][v]
                    )
                    support.add(
                        physical_to_column[p_physical_id]
                    )

                # p_{j+1}(v), when j < r.
                if layer < r:
                    p_next_physical_id = int(
                        primal[layer + 1][v]
                    )
                    support.add(
                        physical_to_column[
                            p_next_physical_id
                        ]
                    )

                # d_j(c) for all M[c,v] = 1.
                start = M_by_column.indptr[v_local]
                end = M_by_column.indptr[v_local + 1]

                neighboring_c_local = (
                    M_by_column.indices[start:end]
                )

                neighboring_checks = []

                for c_local in neighboring_c_local:
                    c = int(C[c_local])
                    neighboring_checks.append(c)

                    d_physical_id = int(
                        dual[layer][c]
                    )
                    support.add(
                        physical_to_column[d_physical_id]
                    )

                for column in sorted(support):
                    row_indices.append(new_row)
                    column_indices.append(column)

                row_metadata.append({
                    "type": "A",
                    "layer": layer,
                    "v": v,
                    "neighboring_checks": neighboring_checks,
                    "support_columns": sorted(support),
                })

                new_row += 1

        values = np.ones(
            len(row_indices),
            dtype=np.uint8,
        )

        added_x_checks = sparse.coo_matrix(
            (
                values,
                (row_indices, column_indices),
            ),
            shape=(new_row, total_columns),
            dtype=np.uint8,
        ).tocsr()

        added_x_checks.data %= 2
        added_x_checks.eliminate_zeros()

        hx_deform = sparse.vstack(
            [
                padded_original_hx,
                added_x_checks,
            ],
            format="csr",
        )

        ancilla_system["new_x_check_metadata"] = row_metadata
        ancilla_system["number_original_x_checks"] = (
            original_hx.shape[0]
        )
        ancilla_system["hx_deform"] = hx_deform

        expected_new_rows = r * len(V)

        if added_x_checks.shape[0] != expected_new_rows:
            raise RuntimeError(
                "Incorrect number of added X checks: "
                f"got {added_x_checks.shape[0]}, "
                f"expected {expected_new_rows}."
            )

        return hx_deform
    
    def new_z_checks(
        self,
        ancilla_system=None,
    ) -> sparse.csr_matrix:
        """
        Construct the complete Z-check matrix Hz_deform for an
        X-type logical deformation.

        The returned matrix contains:

            1. All original BB Z checks.
            2. For c in C, original row c is extended by d_1(c).
            3. New interior checks B_{j,c}, j=2,...,r.

        B_{1,c}:
            H_Z[c, :] plus d_1(c)

        B_{j,c}, j >= 2:
            d_{j-1}(c),
            d_j(c),
            p_j(v) for every v satisfying M[c,v] = 1.
        """
        ancilla_system = (
            self.ancilla_system
            if ancilla_system is None
            else ancilla_system
        )

        if not ancilla_system:
            raise ValueError(
                "No ancilla system has been constructed."
            )

        if ancilla_system.get("basis") != Basis.X:
            raise NotImplementedError(
                "This implementation currently constructs the "
                "X-type logical deformation."
            )

        G1 = ancilla_system["G1"]

        V = np.asarray(
            G1["data_columns"],
            dtype=np.int64,
        ).reshape(-1)

        C = np.asarray(
            G1["check_rows"],
            dtype=np.int64,
        ).reshape(-1)

        M = self._to_binary_csr(
            G1["incidence_matrix"]
        )

        if M.shape != (len(C), len(V)):
            raise ValueError(
                f"M has shape {M.shape}; expected "
                f"({len(C)}, {len(V)})."
            )

        primal = ancilla_system["primal_data"]
        dual = ancilla_system["dual_data"]

        primal_layers = sorted(int(j) for j in primal)
        dual_layers = sorted(int(j) for j in dual)

        r = max(primal_layers)
        expected_layers = list(range(1, r + 1))

        if primal_layers != expected_layers:
            raise ValueError(
                f"Primal layers are {primal_layers}; "
                f"expected {expected_layers}."
            )

        if dual_layers != expected_layers:
            raise ValueError(
                f"Dual layers are {dual_layers}; "
                f"expected {expected_layers}."
            )

        (
            total_columns,
            physical_to_column,
            _,
        ) = self._deformed_data_column_layout(
            ancilla_system
        )

        original_hz = self._to_binary_csr(
            self.code.hz
        )

        original_columns = original_hz.shape[1]

        padding = sparse.csr_matrix(
            (
                original_hz.shape[0],
                total_columns - original_columns,
            ),
            dtype=np.uint8,
        )

        # Begin with all original Z checks.
        modified_original_hz = sparse.hstack(
            [original_hz, padding],
            format="lil",
        )

        modified_row_metadata = []

        # Replace each original row c by:
        #
        #     B_{1,c} = H_Z[c,:] + d_1(c).
        #
        # Since d_1(c) occupies a new column, setting it to 1 is enough.
        for c in C:
            c = int(c)

            d1_physical_id = int(
                dual[1][c]
            )
            d1_column = physical_to_column[
                d1_physical_id
            ]

            modified_original_hz[c, d1_column] = 1

            modified_row_metadata.append({
                "type": "B",
                "layer": 1,
                "original_check_row": c,
                "added_dual_column": d1_column,
            })

        modified_original_hz = (
            modified_original_hz.tocsr()
        )

        # Construct B_{j,c} for j=2,...,r.
        row_indices = []
        column_indices = []
        interior_metadata = []

        new_row = 0

        for layer in range(2, r + 1):
            for c_local, c in enumerate(C):
                c = int(c)
                support = set()

                # d_{j-1}(c)
                d_previous_physical_id = int(
                    dual[layer - 1][c]
                )
                support.add(
                    physical_to_column[
                        d_previous_physical_id
                    ]
                )

                # d_j(c)
                d_current_physical_id = int(
                    dual[layer][c]
                )
                support.add(
                    physical_to_column[
                        d_current_physical_id
                    ]
                )

                # p_j(v) for every M[c,v] = 1.
                neighboring_v_local = (
                    M.getrow(c_local).indices
                )

                neighboring_vertices = []

                for v_local in neighboring_v_local:
                    v = int(V[v_local])
                    neighboring_vertices.append(v)

                    # layer >= 2, so p_j(v) is a new qubit.
                    p_physical_id = int(
                        primal[layer][v]
                    )
                    support.add(
                        physical_to_column[p_physical_id]
                    )

                for column in sorted(support):
                    row_indices.append(new_row)
                    column_indices.append(column)

                interior_metadata.append({
                    "type": "B",
                    "layer": layer,
                    "c": c,
                    "neighboring_vertices": neighboring_vertices,
                    "support_columns": sorted(support),
                })

                new_row += 1

        values = np.ones(
            len(row_indices),
            dtype=np.uint8,
        )

        interior_z_checks = sparse.coo_matrix(
            (
                values,
                (row_indices, column_indices),
            ),
            shape=(new_row, total_columns),
            dtype=np.uint8,
        ).tocsr()

        interior_z_checks.data %= 2
        interior_z_checks.eliminate_zeros()

        hz_deform = sparse.vstack(
            [
                modified_original_hz,
                interior_z_checks,
            ],
            format="csr",
        )

        ancilla_system["modified_z_check_metadata"] = (
            modified_row_metadata
        )
        ancilla_system["new_z_check_metadata"] = (
            interior_metadata
        )
        ancilla_system["number_original_z_checks"] = (
            original_hz.shape[0]
        )
        ancilla_system["hz_deform"] = hz_deform

        expected_new_rows = (r - 1) * len(C)

        if interior_z_checks.shape[0] != expected_new_rows:
            raise RuntimeError(
                "Incorrect number of interior Z checks: "
                f"got {interior_z_checks.shape[0]}, "
                f"expected {expected_new_rows}."
            )

        return hz_deform
    
    def append_deformed_syndrome_round(
        self,
        circ: stim.Circuit,
        hx_deform,
        hz_deform,
        column_to_physical_id,
        x_measure_ancillas,
        z_measure_ancillas,
    ):
        hx = sparse.csr_matrix(hx_deform)
        hz = sparse.csr_matrix(hz_deform)

        # Reset syndrome ancillas.
        circ.append("R", x_measure_ancillas)
        circ.append("R", z_measure_ancillas)

        # Prepare X-check ancillas in |+>.
        circ.append("H", x_measure_ancillas)
        circ.append("TICK")

        # CNOT schedule must be generated here.
        schedule = self.build_deformed_cnot_schedule(
            hx=hx,
            hz=hz,
            column_to_physical_id=column_to_physical_id,
            x_measure_ancillas=x_measure_ancillas,
            z_measure_ancillas=z_measure_ancillas,
        )

        for layer in schedule:
            targets = []

            for control, target in layer:
                targets.extend([control, target])

            if targets:
                circ.append("CX", targets)

            circ.append("TICK")

        # Rotate X-check ancillas back and measure.
        circ.append("H", x_measure_ancillas)

        circ.append(
            "M",
            x_measure_ancillas + z_measure_ancillas,
        )

        return circ
    
    def build_deformed_cnot_schedule(
        self,
        hx,
        hz,
        column_to_physical_id,
        x_measure_ancillas,
        z_measure_ancillas,
    ):
        edges = []

        for row in range(hx.shape[0]):
            ancilla = x_measure_ancillas[row]

            for col in hx.getrow(row).indices:
                data = column_to_physical_id[col]
                edges.append((ancilla, data, "X"))

        for row in range(hz.shape[0]):
            ancilla = z_measure_ancillas[row]

            for col in hz.getrow(row).indices:
                data = column_to_physical_id[col]
                edges.append((data, ancilla, "Z"))

        schedule = []

        for control, target, check_type in edges:
            placed = False

            for tick in schedule:
                occupied = {
                    qubit
                    for c, t in tick
                    for qubit in (c, t)
                }

                if control not in occupied and target not in occupied:
                    tick.append((control, target))
                    placed = True
                    break

            if not placed:
                schedule.append([(control, target)])

        return schedule

    def init_deform_round(self, circ: stim.Circuit, ancilla_system, deformation_system_type: Basis):

        new_data_ids = ancilla_system["new_data_qubit_ids"]
        circ.append("R", new_data_ids)

        if deformation_system_type == Basis.Z:
            circ.append("H", new_data_ids)




class SurfaceCodeAncillaQubit:

    ORDER = ['NW', 'NE', 'SW', 'SE']

    def __init__(self, coord: tuple[float, float], id: int, basis: Basis):
        self.stabilizer_type = basis
        self.coord = coord
        self.id = id
        self.data_qubits: dict[str, DataQubit | None] = {
            direction: None for direction in SurfaceCodeAncillaQubit.ORDER
        }

    def direction_to(self, data_qubit: DataQubit) -> str:
        dx = data_qubit.coord[0] - self.coord[0]
        dy = data_qubit.coord[1] - self.coord[1]

        if dx < 0 and dy > 0:
            return 'NW'
        elif dx > 0 and dy > 0:
            return 'NE'
        elif dx < 0 and dy < 0:
            return 'SW'
        elif dx > 0 and dy < 0:
            return 'SE'
        else:
            raise ValueError(
                f"Data qubit at {data_qubit.coord} is not a neighbor of ancilla at {self.coord}"
            )

    def add_associated_data_qubit(self, data_qubit: DataQubit):
        direction = self.direction_to(data_qubit)
        self.data_qubits[direction] = data_qubit


class SurfaceCodeModule:

    def __init__(self, distance: int, basis: Basis, *, start_idx=0, measurement_book_keeper: "MeasurementBookKeeper" = None):
        self.distance = distance
        self.d = distance
        self.n = distance ** 2
        self.k = 1
        self.basis = basis
        self.start_idx = start_idx

        self.data_qubits = {}
        self.ancilla_qubits = {
            Basis.X: [],
            Basis.Z: [],
        }
        self.x_check_qubits = []
        self.z_check_qubits = []

        self._init_data_qubits()
        self._init_ancillas(Basis.X)
        self._init_ancillas(Basis.Z)

        all_ids = [
            q.id
            for q in list(self.data_qubits.values())
            + self.x_check_qubits
            + self.z_check_qubits
        ]
        if measurement_book_keeper is not None:
            self.measurement_book_keeper = measurement_book_keeper
            self.measurement_book_keeper.register_qubits(all_ids)
        else:
            self.measurement_book_keeper = MeasurementBookKeeper(qubits=all_ids)

    def _init_data_qubits(self):
        for x in range(self.distance):
            for y in range(self.distance):
                self.data_qubits[(x, y)] = DataQubit((x, y), self.start_idx + x * self.distance + y, None)

        self.num_qubits = self.start_idx + self.distance ** 2

    def _init_ancillas(self, basis: Basis):
        transpose = basis == Basis.Z

        def _get_neighbors(r, c, direction):
            coords = [(r, c), (r + 1, c)]
            col_shift = -1 if direction else 1
            coords.extend([(r, c + col_shift), (r + 1, c + col_shift)])
            return coords

        ancillas = []
        direction = False
        for i in range(self.distance - 1):
            for j in range(0, self.distance, 2):
                current_direction = direction if not transpose else not direction
                plaquette_coords = _get_neighbors(i, j, current_direction)

                if transpose:
                    plaquette_coords = [(c, r) for r, c in plaquette_coords]

                xs = [c[0] for c in plaquette_coords]
                ys = [c[1] for c in plaquette_coords]
                ancilla_coord = (sum(xs) / len(xs), sum(ys) / len(ys))

                anc = SurfaceCodeAncillaQubit(ancilla_coord, self.num_qubits, basis)
                self.num_qubits += 1

                for coord in plaquette_coords:
                    if coord in self.data_qubits:
                        anc.add_associated_data_qubit(self.data_qubits[coord])

                ancillas.append(anc)
            direction = not direction

        if basis == Basis.X:
            self.x_check_qubits = ancillas
        else:
            self.z_check_qubits = ancillas
        self.ancilla_qubits[basis] = ancillas

    def initial_round(self):
        circ = stim.Circuit()

        def _qubit_coord_text(qubits):
            text = ""
            for q in qubits:
                text += f"QUBIT_COORDS(0, {q.id}) {q.id}\n"
            return text

        circ.append_from_stim_program_text(_qubit_coord_text(self.x_check_qubits))
        circ.append_from_stim_program_text(_qubit_coord_text(list(self.data_qubits.values())))
        circ.append_from_stim_program_text(_qubit_coord_text(self.z_check_qubits))

        all_qubits = self.x_check_qubits + list(self.data_qubits.values()) + self.z_check_qubits
        circ.append("R", [q.id for q in all_qubits])
        if self.basis == Basis.X:
            circ.append("H", [q.id for q in self.data_qubits.values()])
        circ.append("TICK")
        return circ

    def _yield_a_stage_syndrome_ex(self, order: int):
        x_directions = ['NW', 'NE', 'SW', None, 'SE']
        z_directions = ['NW', None, 'SW', 'NE', 'SE']

        def get_interaction_pair(anc: SurfaceCodeAncillaQubit, direction):
            if direction is None:
                return None, None
            data_qubit = anc.data_qubits[direction]
            if data_qubit is None:
                return None, None
            if anc.stabilizer_type == Basis.X:
                return anc.id, data_qubit.id
            elif anc.stabilizer_type == Basis.Z:
                return data_qubit.id, anc.id

        x_ctrls, x_targs, z_ctrls, z_targs = [], [], [], []

        for anc in self.x_check_qubits:
            ctrl, targ = get_interaction_pair(anc, x_directions[order])
            if ctrl is not None and targ is not None:
                x_ctrls.append(ctrl)
                x_targs.append(targ)

        for anc in self.z_check_qubits:
            ctrl, targ = get_interaction_pair(anc, z_directions[order])
            if ctrl is not None and targ is not None:
                z_ctrls.append(ctrl)
                z_targs.append(targ)

        return x_ctrls, x_targs, z_ctrls, z_targs

    def syndrome_extraction(self, circ: stim.Circuit):
        circ.append("H", [q.id for q in self.x_check_qubits])

        for order in range(5):
            x_ctrls, x_targs, z_ctrls, z_targs = self._yield_a_stage_syndrome_ex(order)

            pairs = []
            for ct in zip(x_ctrls, x_targs):
                pairs.append(ct[0])
                pairs.append(ct[1])
            for ct in zip(z_ctrls, z_targs):
                pairs.append(ct[0])
                pairs.append(ct[1])

            circ.append("CX", pairs)
            circ.append("TICK")

        circ.append("H", [q.id for q in self.x_check_qubits])
        return circ

    def measurement(self, circ, qubits):
        measurement_q_ids = [q.id for q in qubits]
        circ.append("MR", measurement_q_ids)

        for q_id in measurement_q_ids:
            self.measurement_book_keeper.record_measurement(q_id)

        return circ

    def checks_detection_events(self, checks):
        instructions = []
        for q in checks:
            event = self.measurement_book_keeper.get_detection_event(q.id)

            if isinstance(event, int):
                instr = stim.CircuitInstruction("DETECTOR", [stim.target_rec(event)], (0, 0, q.id))
                instructions.append(instr)
            elif isinstance(event, tuple) or isinstance(event, list):
                instr = stim.CircuitInstruction("DETECTOR", [stim.target_rec(event[0]), stim.target_rec(event[1])], (0, 0, q.id))
                instructions.append(instr)

        return instructions

    def qec_round(self, circ: stim.Circuit, detector_basis: list[Basis]):
        circ = self.syndrome_extraction(circ)
        circ = self.measurement(circ, self.x_check_qubits + self.z_check_qubits)

        checks = []
        if Basis.X in detector_basis:
            checks += self.x_check_qubits
        if Basis.Z in detector_basis:
            checks += self.z_check_qubits

        instructions = self.checks_detection_events(checks)
        for instr in instructions:
            circ.append(instr)

        circ.append("SHIFT_COORDS", [], [1, 0, 0])
        return circ

    def measure_all_data_qubits(self, circ):
        data_qubits = list(self.data_qubits.values())
        if self.basis == Basis.X:
            circ.append("H", [q.id for q in data_qubits])

        return self.measurement(circ, data_qubits)

    def _logical_observable(self):
        involved_qubits = []
        for x in range(self.distance):
            coord = (x, 0) if self.basis == Basis.Z else (0, x)
            if coord in self.data_qubits:
                involved_qubits.append(self.data_qubits[coord].id)
        yield involved_qubits

    def observable(self, circ):
        for idx, ops in enumerate(self._logical_observable()):
            targets = []
            for q_id in ops:
                targets.append(self.measurement_book_keeper.get_detection_event(q_id))
            circ.append("OBSERVABLE_INCLUDE", [stim.target_rec(t) for t in targets], idx)

        return circ

    def finalize(self, circ: stim.Circuit):
        checks = self.x_check_qubits if self.basis == Basis.X else self.z_check_qubits

        for anc in checks:
            targets = []
            for direction, neighbor in anc.data_qubits.items():
                if neighbor is not None:
                    targets.append(
                        self.measurement_book_keeper.get_last_measurement(neighbor.id)
                        - self.measurement_book_keeper.measurement_count
                    )
            targets.append(
                self.measurement_book_keeper.get_last_measurement(anc.id)
                - self.measurement_book_keeper.measurement_count
            )
            targets = [stim.target_rec(t) for t in targets]
            circ.append("DETECTOR", targets, (0, 0, anc.id))

        return circ

    def get_logical_support(self, basis: Basis) -> list[int]:
        """Qubit ids for the logical operator of the given basis, regardless
        of self.basis (needed to reach the "other" logical operator, e.g.
        getting Z support out of an X-memory module for a joint measurement)."""
        involved = []
        for x in range(self.distance):
            coord = (x, 0) if basis == Basis.Z else (0, x)
            if coord in self.data_qubits:
                involved.append(self.data_qubits[coord].id)
        return involved


class Deformation:
    """
    Code deformation / logical LOAD: teleports the logical state of one
    BB-code logical qubit into a fresh surface code patch, following the
    qSIEVE LD/ST protocol (Viszlai et al., arXiv:2311.16980; ancilla
    construction from Cohen et al., Science Advances 8, eabn1717 (2022)):

        1. Prepare the destination (surface code) in |+_L>.
        2. Fault-tolerantly measure the joint logical operator Z_A Z_B.
        3. Measure out the source (BB code) logical qubit in the X basis.
        4. Apply a Pauli correction determined by the two outcomes (tracked
           in software -- folded directly into how the observable is
           defined, no physical gate needed).

    Simplification: the paper mediates the joint Z_A Z_B measurement with a
    dedicated hypergraph-product ancilla system with its own spatial
    redundancy. Here it's measured directly via a single native Pauli
    product measurement (MPP) spanning both logical operators' physical
    supports, optionally repeated `bridge_copies` times with detectors
    tying consecutive copies together for redundancy against a single bad
    measurement. That's a real simplification of the full construction
    (Section 9.4 of Bravyi et al.), but the joint measurement itself is
    physically exact, not approximated.

    Two things worth knowing, found by hitting stim's own determinism
    checker (`circuit.detector_error_model()` refuses to build a DEM for a
    circuit whose observables/detectors aren't fully determined by the
    recorded measurements -- an invaluable correctness check that caught
    both of these):

    1. bposd's `code.lx[i]` / `code.lz[i]` are NOT guaranteed to be a
       matched anticommuting pair for the same index `i` -- they're just
       independent bases for the X-logical and Z-logical spaces. Using
       lz[logical_index] directly as "the" partner of lx[logical_index]
       silently picks two operators that may commute, which breaks the
       teleportation math outright. __init__ finds lz[i]'s *actual*
       anticommuting partner via the symplectic inner product before
       building anything.

    2. Z_A Z_B genuinely does not commute with X_A (that back-action is
       what makes teleportation work at all), so any detector that spans
       across the bridge measurement -- i.e. depends on data qubits both
       before and after it -- can become impossible for stim to resolve
       deterministically. BBCodeModule.finalize() is exactly such a
       detector (it reconciles the last pre-bridge X-check round against
       the post-bridge data read-out), so it is deliberately *not* called
       on the source after the bridge here. This sacrifices catching
       errors in the final window on the source side specifically (the
       source's data is being destroyed by the teleportation anyway); the
       destination's own finalize() is unaffected and still runs.
    """

    def __init__(self, code: BivariateBicycleCode, surface_distance: int, *, logical_index: int = 0):
        self.logical_index = logical_index
        self.book_keeper = MeasurementBookKeeper()

        self.bb = BBCodeModule(code, Basis.Z, measurement_book_keeper=self.book_keeper)

        bb_ids = [
            q.id for q in
            self.bb.l_data_qubits + self.bb.r_data_qubits + self.bb.x_check_qubits + self.bb.z_check_qubits
        ]
        surface_start_idx = max(bb_ids) + 1

        self.surface = SurfaceCodeModule(
            surface_distance, Basis.X,
            start_idx=surface_start_idx,
            measurement_book_keeper=self.book_keeper,
        )

        self.bb_z_index = self._find_anticommuting_partner(code, logical_index)
        self.bb_z_support = self.bb.get_logical_z_support(self.bb_z_index)
        self.surface_z_support = self.surface.get_logical_support(Basis.Z)

    @staticmethod
    def _find_anticommuting_partner(code: BivariateBicycleCode, x_index: int) -> int:
        """bposd's lx/lz rows aren't guaranteed paired by matching index --
        find which lz row actually anticommutes with lx[x_index] via the
        symplectic inner product (shared-support parity)."""
        lx_row = np.asarray(code.lx[x_index].todense()).flatten().astype(int)
        for j in range(code.k):
            lz_row = np.asarray(code.lz[j].todense()).flatten().astype(int)
            if int((lx_row & lz_row).sum()) % 2 == 1:
                return j
        raise ValueError(f"No lz row anticommutes with lx[{x_index}]; code.lx/lz may not form a valid symplectic basis.")

    def joint_zz_measurement(self, circ: stim.Circuit, bridge_copies: int = 1) -> stim.Circuit:
        """Fault-tolerantly measures Z_A Z_B via a native Pauli product
        measurement (MPP), repeated `bridge_copies` times for redundancy.
        Each repetition is a genuine re-measurement of the now-fixed joint
        eigenvalue (safe to repeat, like any stabilizer), not a
        re-coupling via a fresh ancilla, so there's no reset-related
        back-action buildup."""
        pauli_targets = []
        for q_id in self.bb_z_support + self.surface_z_support:
            pauli_targets.append(stim.target_z(q_id))
            pauli_targets.append(stim.target_combiner())
        pauli_targets = pauli_targets[:-1]

        mpp_recs = []
        for _ in range(bridge_copies):
            circ.append("MPP", pauli_targets)
            mpp_recs.append(self.book_keeper.measurement_count)
            self.book_keeper.measurement_count += 1

        for i in range(1, len(mpp_recs)):
            targets = [
                stim.target_rec(mpp_recs[i - 1] - self.book_keeper.measurement_count),
                stim.target_rec(mpp_recs[i] - self.book_keeper.measurement_count),
            ]
            circ.append("DETECTOR", targets, (0, 0, 9_000_000 + i))
        if bridge_copies > 1:
            circ.append("SHIFT_COORDS", [], [1, 0, 0])
        return circ

    def observable(self, circ: stim.Circuit) -> stim.Circuit:
        """Final logical observable = (source's logical-X read-out) XOR
        (destination's logical-X read-out). If teleportation succeeds and
        both started in |+_L>, this is deterministically 0 in the
        noiseless case: the X-type correction from the joint-ZZ outcome
        commutes with an X-basis read-out of the destination (so it's
        omitted here), and the Z-type correction from the source's X-basis
        read-out is exactly what this XOR already cancels against."""
        bb_x_support = self.bb.get_logical_x_support(self.logical_index)
        surface_x_support = self.surface.get_logical_support(Basis.X)

        targets = [
            self.bb.measurement_book_keeper.get_detection_event(q_id)
            for q_id in bb_x_support
        ] + [
            self.surface.measurement_book_keeper.get_detection_event(q_id)
            for q_id in surface_x_support
        ]
        circ.append("OBSERVABLE_INCLUDE", [stim.target_rec(t) for t in targets], self.logical_index)
        return circ

    def build_circuit(self, bb_rounds: int, surface_rounds: int, bridge_copies: int = 1) -> stim.Circuit:
        """Full LOAD circuit: protect the source in memory for bb_rounds,
        fault-tolerantly transfer via the LD/ST bridge, measure out the
        source, then protect the destination in compute for surface_rounds
        before reading it out."""
        circ = self.bb.initial_round()
        circ += self.surface.initial_round()

        circ = self.bb.qec_round(circ, detector_basis=[Basis.X])
        for _ in range(bb_rounds - 1):
            circ = self.bb.qec_round(circ, detector_basis=[Basis.X, Basis.Z])

        circ = self.joint_zz_measurement(circ, bridge_copies)

        circ = self.bb.measure_all_data_qubits(circ)
        # NOTE: bb.finalize() is deliberately skipped here -- see class docstring point 2.

        circ = self.surface.qec_round(circ, detector_basis=[Basis.X])
        for _ in range(surface_rounds - 1):
            circ = self.surface.qec_round(circ, detector_basis=[Basis.X, Basis.Z])

        circ = self.surface.measure_all_data_qubits(circ)
        circ = self.surface.finalize(circ)

        circ = self.observable(circ)
        return circ


if __name__ == '__main__':

    bbcode = BBCodeModule(
        code=BivariateBicycleCode.generate_code(distance=12),
        basis=Basis.X
    )

    involved, check_indices, M = bbcode.ancilla_system_for_one_logical_operator(
        bbcode.code.lx, 0, bbcode.code.hz
    )

    row_parities = np.asarray(M.sum(axis=1)).ravel() % 2
    assert np.all(row_parities == 0), (
        "Every neighboring opposite-type check must overlap the logical "
        "operator an even number of times."
    )
    poly_f = Polynomial("1 + x + x*y^3 + x^2 + x^3 + x^5*y^3 + x^6 + x^7 + x^7*y^3 + x^8 +x^9 + x^11*y^3") 
    poly_g = Polynomial(terms=[(1, 0), (2, 1), (0, 2), (1, 2), (2, 3), (0, 4)])
    poly_h = Polynomial(terms=[(0, 0), (0, 1), (1, 1), (0, 2), (0, 3), (1, 3)])
    ancilla_system = bbcode.build_ancilla_system(poly_f, Polynomial("0"))
    hx_deform = bbcode.new_x_checks(ancilla_system)
    hz_deform = bbcode.new_z_checks(ancilla_system)

    assert np.all((hx_deform @ hz_deform.T).toarray() % 2 == 0), """Anticommutive stabilizer"""

    circ = bbcode.initial_round()
    circ = bbcode.qec_round(circ, detector_basis=[Basis.X])
    


    # bbcode.reset_ancilla_system()
    # ancilla_system = bbcode.build_ancilla_system(poly_g, poly_h)
    # hx_deform = bbcode.new_x_checks(ancilla_system)
    # hz_deform = bbcode.new_z_checks(ancilla_system)

    # assert np.all((hx_deform @ hz_deform.T).toarray() % 2 == 0), """Anticommutive stabilizer"""
    

    # deformation = Deformation(
    #     code=BivariateBicycleCode.generate_code(distance=12),
    #     surface_distance=5,
    # )

    # circ = deformation.build_circuit(bb_rounds=12, surface_rounds=5, bridge_copies=1)
    # circ.to_file("deformation_test.stim")
    # noise_profile = NoiseModel(
    #     gate_profile=GateProfile(),
    #     after_clifford_error=DepolarizingNoise(fixed_noise_level=1e-3),
    #     before_measurement_error=FlipNoise(fixed_noise_level=1e-3),
    #     after_reset_error=FlipNoise(fixed_noise_level=1e-3),
    # )
    # noisy_circ = noise_profile.apply(circ)
    # noisy_circ.to_file("deformation_test_noisy.stim")
    # print(noisy_circ.detector_error_model())

    # code = BivariateBicycleCode.generate_code(distance=12)
    # print(f"n: {code.n}, k: {code.k}")


    # print(code.hx)
    # print(code.hz)


    # code_module = BBCodeModule(code, Basis.X)
    # circ = code_module.initial_round()
    # circ = code_module.qec_round(circ, detector_basis=[Basis.X])

    # repeated_block = stim.Circuit()
    # repeated_block = code_module.qec_round(repeated_block, detector_basis=[Basis.X])
    # repeated_block = stim.CircuitRepeatBlock(code.d - 1, repeated_block)
    # circ.append(repeated_block)
    # circ = code_module.measure_all_data_qubits(circ)
    # circ = code_module.finalize(circ)
    # circ = code_module.observable(circ)
    # circ.to_file("test.stim")
    # dem = circ.detector_error_model()
    # print(dem)


    # # generate circuit

    # tasks = []
    # for p in [1e-2, 5e-3, 2e-3, 1e-3]:
    #     print(f"Simulating with error rate: {p}")

    #     code_module = BBCodeModule(code, Basis.X)
    #     circ = code_module.initial_round()
    #     circ = code_module.qec_round(circ, detector_basis=[Basis.X])

    #     repeated_block = stim.Circuit()
    #     repeated_block = code_module.qec_round(repeated_block, detector_basis=[Basis.X])
    #     repeated_block = stim.CircuitRepeatBlock(code.d - 1, repeated_block)
    #     circ.append(repeated_block)
    #     circ = code_module.measure_all_data_qubits(circ)
    #     circ = code_module.finalize(circ)
    #     circ = code_module.observable(circ)


    #     noise_profile = NoiseModel(
    #         gate_profile=GateProfile(),
    #         after_clifford_error=DepolarizingNoise(fixed_noise_level=p),
    #         before_measurement_error=FlipNoise(fixed_noise_level=p),
    #         after_reset_error=FlipNoise(fixed_noise_level=p)
    #     )

    #     noisy_circuit = noise_profile.apply(circ)
    #     noisy_circuit.to_file(f"noisy_circuit_p_{p}.stim")

    #     tasks.append(
    #         sinter.Task(
    #             circuit=noisy_circuit,
    #             json_metadata={"p": p}
    #         )
    #     )