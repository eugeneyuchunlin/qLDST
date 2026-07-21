import re
import stim
import numpy as np
from galois import GF
from bposd.css import css_code
from ldpc import SinterBpOsdDecoder

try:
    from error import NoiseModel, DepolarizingNoise, FlipNoise, IdlingNoise
    from gate_profile import GateProfile
    from utils.position import Vec2
    from patch import (
        DataQubit,
        AncillaQubit,
        CodeModule,
        CoarseIdlingMixin,
        ScatteredIdlingMixin,
        FineGrainedIdlingMixin,
    )
    from basis import Basis
    from layers import *
    from utils.visualization import draw_qubit_gantt
except ImportError:
    from .error import NoiseModel, DepolarizingNoise, FlipNoise, IdlingNoise
    from .gate_profile import GateProfile
    from .utils.position import Vec2
    from .patch import (
        DataQubit,
        AncillaQubit,
        CodeModule,
        CoarseIdlingMixin,
        ScatteredIdlingMixin,
        FineGrainedIdlingMixin,
    )
    from .basis import Basis
    from .layers import *
    from .utils.visualization import draw_qubit_gantt


class Polynomial:
    def __init__(self, polynomial_str: str):

        self.terms = self._parse_polynomial(polynomial_str)

    def _parse_polynomial(self, polynomial_str: str):
        monomials = polynomial_str.replace(" ", "").split("+")

        results = []

        for m in monomials:
            x_match = re.search(r"x(?:\^(\d+))?", m)
            x_exp = 0
            if x_match:
                x_exp = int(x_match.group(1)) if x_match.group(1) else 1

            y_match = re.search(r"y(?:\^(\d+))?", m)
            y_exp = 0
            if y_match:
                y_exp = int(y_match.group(1)) if y_match.group(1) else 1

            if not x_match and not y_match and m == "1":
                x_exp, y_exp = 0, 0

            results.append((x_exp, y_exp))

        return results

    def __repr__(self):
        return f"Polynomial(terms={self.terms})"


class BBAncillaQubit(AncillaQubit):
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
        stabilizer_type = basis
        self.poly_a_terms = poly_a_terms
        self.poly_b_terms = poly_b_terms
        self.l = l
        self.m = m

        super().__init__(coord, id, stabilizer_type)

    def add_associated_data_qubits(self, data_qubit: list[DataQubit]):

        for i in range(len(data_qubit)):
            self.data_qubits[i] = data_qubit[i]

    def direction_to(self, data_qubit):
        pass


class BivariateBicycleCode:

    def __init__(self, l, m, poly_a: Polynomial, poly_b: Polynomial):
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
        self.d = None

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
            )
        else:
            raise NotImplementedError(f"Distance: {distance} is not supported.")


class BivariateBicycleCodePatch(CodeModule):

    def __init__(
        self,
        bbcode: BivariateBicycleCode,
        rounds: int,
        basis: Basis,
        gateset="CX",
    ):

        self.bb = bbcode
        self.x_check_indices = []
        self.l_data_indices = []
        self.r_data_indices = []
        self.z_check_indices = []

        super().__init__(
            distance=self.bb.d,
            rounds=rounds,
            basis=basis,
            gateset=gateset,
        )

    def _setup_geometry(self):
        pass

    def _init_data_qubits(self):

        # L: 0-71
        # R: 72-143

        offset = self.bb.n // 2

        for _id in range(self.bb.n):
            i, j = divmod(_id, self.bb.l)
            if _id < offset:
                self.l_data_indices.append(_id + offset)
            else:
                self.r_data_indices.append(_id + offset)
            self.data_qubits[_id + offset] = DataQubit(
                coord=(0, _id), id=_id + offset, basis=self.basis
            )
        self.num_qubits = len(self.data_qubits)

    def _post_init(self):
        return self.reset_circuit()

    def _init_ancillas(self, basis: Basis):
        coord_second_part_offset = 0 if basis == Basis.X else self.bb.n // 2
        id_offset = 0 if basis == Basis.X else 3 * self.bb.n // 2
        data_qubit_offset = self.bb.n // 2

        poly_a = self.bb.poly_a
        poly_b = self.bb.poly_b

        ancillas = []
        for i in range(self.bb.l):
            for j in range(self.bb.m):

                if basis == Basis.X:
                    # 1. Forward shifts for X
                    terms_a = [
                        ((i + t[0]) % self.bb.l, (j + t[1]) % self.bb.m)
                        for t in poly_a.terms
                    ]
                    terms_b = [
                        ((i + t[0]) % self.bb.l, (j + t[1]) % self.bb.m)
                        for t in poly_b.terms
                    ]

                    # 2. A -> Left Data (offset 0), B -> Right Data (offset n_data)
                    A_targets = [
                        self.data_qubits[t[0] * self.bb.m + t[1] + data_qubit_offset]
                        for t in terms_a
                    ]
                    B_targets = [
                        self.data_qubits[
                            t[0] * self.bb.m + t[1] + 2 * data_qubit_offset
                        ]
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
                        ((i - t[0]) % self.bb.l, (j - t[1]) % self.bb.m)
                        for t in poly_a.terms
                    ]
                    terms_b = [
                        ((i - t[0]) % self.bb.l, (j - t[1]) % self.bb.m)
                        for t in poly_b.terms
                    ]

                    # 2. A^T -> Right Data (offset n_data), B^T -> Left Data (offset 0)
                    A_targets = [
                        self.data_qubits[
                            t[0] * self.bb.m + t[1] + 2 * data_qubit_offset
                        ]
                        for t in terms_a
                    ]
                    B_targets = [
                        self.data_qubits[t[0] * self.bb.m + t[1] + data_qubit_offset]
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
                    coord=(1, i * self.bb.m + j + coord_second_part_offset),
                    id=id_offset + i * self.bb.m + j,
                    basis=basis,
                    l=self.bb.l,
                    m=self.bb.m,
                    poly_a_terms=terms_a,
                    poly_b_terms=terms_b,
                )

                anc.add_associated_data_qubits(ordered)
                ancillas.append(anc)
        if basis == Basis.X:
            self.x_check_indices = [anc.id for anc in ancillas]
        else:
            self.z_check_indices = [anc.id for anc in ancillas]
        self.ancilla_qubits[basis] = ancillas
        self.num_qubits += len(ancillas)

    def _logical_observable(self):

        # logical_matrix = self.bb.lz if self.basis == Basis.Z else self.bb.lx
        if self.basis.value == Basis.Z.value:
            logical_matrix = self.bb.lz
        elif self.basis.value == Basis.X.value:
            logical_matrix = self.bb.lx
        else:
            raise ValueError(f"Basis {self.basis} not supported")

        n = self.bb.n

        for obs_indx, logical_op in enumerate(logical_matrix):
            involved_qubits = np.nonzero(logical_op)[1]
            # print(involved_qubits)
            yield involved_qubits

    def _add_logical_observable(self, data_meas_map):
        observable_layers = []
        for idx, obs in enumerate(self._logical_observable()):
            targets = []
            for q_id in obs:
                targets.append(data_meas_map[q_id + self.bb.n // 2])
            observable_layers.append(
                ObservableLayer("OBSERVABLE_INCLUDE", targets, idx)
            )
        return observable_layers

    def _yield_a_stage_syndrome_ex(
        self, stage: int, x_basis_ancillas, z_basis_ancillas
    ):
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
            for anc in z_basis_ancillas:
                ctrl, targ, op = self._get_interaction_pair(anc, z_stage)
                if ctrl is not None and targ is not None:
                    ops[Basis.Z]["control"].append(ctrl)
                    ops[Basis.Z]["target"].append(targ)

        if x_stage >= 0:
            for anc in x_basis_ancillas:
                ctrl, targ, op = self._get_interaction_pair(anc, x_stage)
                if ctrl is not None and targ is not None:
                    ops[Basis.X]["control"].append(ctrl)
                    ops[Basis.X]["target"].append(targ)

        return (
            ops[Basis.X]["control"],
            ops[Basis.X]["target"],
            ops[Basis.Z]["control"],
            ops[Basis.Z]["target"],
        )

    def _yield_syndrome_routing(self, all_ancillas: list["AncillaQubit"]):
        z_basis_ancillas = [
            anc for anc in all_ancillas if anc.stabilizer_type == Basis.Z
        ]
        x_basis_ancillas = [
            anc for anc in all_ancillas if anc.stabilizer_type == Basis.X
        ]

        for r in range(7):
            yield self._yield_a_stage_syndrome_ex(r, z_basis_ancillas, x_basis_ancillas)

    def _separate_part(self, layers, data_checks):
        _sub_layers = []
        for layer in layers:
            if isinstance(layer, GateLayer):
                layer_qubits = set(layer.qubits)
                if data_checks & layer_qubits:
                    _sub_layers.append(layer)
        return _sub_layers

    def compose(self, circ: stim.Circuit, gate_layers: list[GateLayer]):
        super().compose(circ, gate_layers)

    def qec_round(self, half_detector: bool = False):
        # print(f"qec round, detector is : {half_detector}")
        circ = stim.Circuit()
        gate_layers = []
        gate_layers += self.ancilla_hadamard_layer(stabilizer_type=Basis.Z)

        # Round 0
        X_controls, X_targets, Z_controls, Z_targets = self._yield_a_stage_syndrome_ex(
            0, self.ancilla_qubits[Basis.X], self.ancilla_qubits[Basis.Z]
        )
        cnot_layer = self._gateset_multiplex(
            X_controls, X_targets, Z_controls, Z_targets
        )
        hadamard_layer = self.ancilla_hadamard_layer(stabilizer_type=Basis.X)
        gate_layers.append(MixedLayer([cnot_layer, hadamard_layer]))

        # Round 1-5
        for stage in range(1, 6):
            X_controls, X_targets, Z_controls, Z_targets = (
                self._yield_a_stage_syndrome_ex(
                    stage, self.ancilla_qubits[Basis.X], self.ancilla_qubits[Basis.Z]
                )
            )
            gate_layers += self._gateset_multiplex(
                X_controls, X_targets, Z_controls, Z_targets
            )

        # Round 6
        X_controls, X_targets, Z_controls, Z_targets = self._yield_a_stage_syndrome_ex(
            6, self.ancilla_qubits[Basis.X], self.ancilla_qubits[Basis.Z]
        )

        gate_layers += self._gateset_multiplex(
            X_controls, X_targets, Z_controls, Z_targets
        )
        gate_layers += self.ancilla_hadamard_layer(stabilizer_type=Basis.Z)
        gate_layers += self.ancilla_hadamard_layer(stabilizer_type=Basis.X)
        # gate_layers += self.measure_and_add_detectors(self.ancilla_qubits[Basis.Z], half_detector)

        # Round 7
        gate_layers += self.measure_and_add_detectors(
            self.ancilla_qubits[Basis.X] + self.ancilla_qubits[Basis.Z], half_detector
        )
        return gate_layers


class CoarseIdlingBivariateBicycleCodePatch(
    BivariateBicycleCodePatch, CoarseIdlingMixin
):
    pass


class ScatteredIdlingBivariateBicycleCodePatch(
    BivariateBicycleCodePatch, ScatteredIdlingMixin
):
    pass


class FineGrainedIdlingBivariateBicycleCodePatch(
    BivariateBicycleCodePatch, FineGrainedIdlingMixin
):

    def idled_syndrome_extraction(
        self,
        all_ancillas: list[AncillaQubit],
        idling_time: float,
        half_detector: bool = True,
    ):
        gate_layers = []
        each_stage_idling_time = idling_time / 6
        print("each_stage_idling_time: ", each_stage_idling_time)

        gate_layers += self.ancilla_hadamard_layer(stabilizer_type=Basis.Z)

        # Round 0
        q_ids = [q for q in list(self.r_data_indices) + list(self.z_check_indices)]
        gate_layers.append(IdlingLayer("idling_layer", q_ids, each_stage_idling_time))
        X_controls, X_targets, Z_controls, Z_targets = self._yield_a_stage_syndrome_ex(
            0, self.ancilla_qubits[Basis.X], self.ancilla_qubits[Basis.Z]
        )
        gate_layers += self._gateset_multiplex(
            X_controls, X_targets, Z_controls, Z_targets
        )
        gate_layers += self.ancilla_hadamard_layer(stabilizer_type=Basis.X)

        # Round 1-5
        for stage in range(1, 6):
            q_ids = [
                q.id
                for q in list(self.data_qubits.values())
                + self.ancilla_qubits[Basis.X]
                + self.ancilla_qubits[Basis.Z]
            ]
            gate_layers.append(
                IdlingLayer("idling_layer", q_ids, each_stage_idling_time)
            )
            # self.add_idling_time(each_stage_idling_time, q_ids)
            X_controls, X_targets, Z_controls, Z_targets = (
                self._yield_a_stage_syndrome_ex(
                    stage, self.ancilla_qubits[Basis.X], self.ancilla_qubits[Basis.Z]
                )
            )
            gate_layers += self._gateset_multiplex(
                X_controls, X_targets, Z_controls, Z_targets
            )

        # Round 6
        q_ids = [q for q in list(self.l_data_indices) + list(self.x_check_indices)]
        gate_layers.append(IdlingLayer("idling_layer", q_ids, each_stage_idling_time))

        X_controls, X_targets, Z_controls, Z_targets = self._yield_a_stage_syndrome_ex(
            6, self.ancilla_qubits[Basis.X], self.ancilla_qubits[Basis.Z]
        )

        gate_layers += self._gateset_multiplex(
            X_controls, X_targets, Z_controls, Z_targets
        )
        gate_layers += self.ancilla_hadamard_layer(stabilizer_type=Basis.Z)
        # gate_layers += self.measure_and_add_detectors(self.ancilla_qubits[Basis.Z], half_detector)

        # Round 7
        gate_layers += self.ancilla_hadamard_layer(stabilizer_type=Basis.X)
        gate_layers += self.measure_and_add_detectors(
            self.ancilla_qubits[Basis.X] + self.ancilla_qubits[Basis.Z], half_detector
        )
        return gate_layers

    def generate_idling_circuit(self, idling_time, half_detector=False):
        if idling_time <= 0:
            return self.circuit(half_detector=half_detector)

        gate_layers = []
        gate_layers += self.initialize_all_qubits()
        gate_layers += self.qec_round(True)
        gate_layers = self.optimize(gate_layers)
        self.compose(self.circ, gate_layers)

        repeat_gate_layers = self.idled_syndrome_extraction(
            all_ancillas=self.ancilla_qubits[Basis.Z] + self.ancilla_qubits[Basis.X],
            idling_time=idling_time,
        )

        repeat_block_circuit = stim.Circuit()
        repeat_gate_layers = self.optimize(repeat_gate_layers)
        self.compose(repeat_block_circuit, repeat_gate_layers)
        self.circ.append(stim.CircuitRepeatBlock(self.rounds - 1, repeat_block_circuit))

        final_layer = self.measure_and_finalize()
        self.compose(self.circ, final_layer)

        return self.circ


if __name__ == "__main__":
    from utils.helper import view_circuit_timeline_diagram
    from gate_profile import Gate

    bb = BivariateBicycleCode.generate_code(12)

    bbcode_patch = BivariateBicycleCodePatch(
        bbcode=bb,
        rounds=24,
        basis=Basis.Z,
        gateset="CX",
    )
    circuit = bbcode_patch.circuit(half_detector=True)
    # q = bbcode_patch.ancilla_qubits[Basis.Z][0]
    # print(q.id)
    # for entry in q.timeline:
    #     print(entry)
    circuit.to_file("bb.stim")
    dem = circuit.detector_error_model()
    print(dem)
    exit(0)

    gate_profile = GateProfile.from_string(
        "CNOT,1.00e-07,1e-03;CZ,1.00e-07,1e-03;H,1.00e-07,1e-03;MR,1e-03,1e-03;M,1e-03,1e-03;R,1e-03,1.00e-03"
    )
    bbcode_patch = FineGrainedIdlingBivariateBicycleCodePatch(
        bbcode=bb,
        rounds=120,
        basis=Basis.X,
        gateset="CZ",
        error_profile=NoiseModel(
            idling_error=IdlingNoise(
                basis=Basis.X, idle_multiplier=1, default_T1=4, default_T2=1
            ),
            before_measurement_error=FlipNoise(),
            after_measurement_error=FlipNoise(),
            after_clifford_error=DepolarizingNoise(),
            gate_profile=gate_profile,
        ),
        gate_profile=gate_profile,
    )

    circuit = bbcode_patch.generate_idling_circuit(1, half_detector=True)
    # fig, ax = draw_qubit_gantt(
    #     bbcode_patch,
    #     qubit_ids=[
    #         bbcode_patch.ancilla_qubits[Basis.X][0].id,
    #         bbcode_patch.l_data_indices[0],
    #         bbcode_patch.r_data_indices[0],
    #         bbcode_patch.ancilla_qubits[Basis.Z][0].id,
    #     ],
    #     include_idle=True,
    #     label_entries=True,
    #     time_unit="us",
    #     time_scale="compressed",
    #     save_path="qubit_gantt.png",
    #     show=True,
    # )
    circuit.detector_error_model()
    circuit.to_file("simple_bb.stim")
    # view_circuit_timeline_diagram(circuit)
    # exit(0)

    bbcode_patch = ScatteredIdlingBivariateBicycleCodePatch(
        bbcode=bb,
        rounds=120,
        basis=Basis.X,
        gateset="CX",
        error_profile=NoiseModel(
            idling_error=IdlingNoise(
                basis=Basis.X, idle_multiplier=1, default_T1=4, default_T2=1
            ),
            before_measurement_error=FlipNoise(),
            after_measurement_error=FlipNoise(),
            after_clifford_error=DepolarizingNoise(),
            gate_profile=gate_profile,
        ),
        gate_profile=gate_profile,
    )
    print(f"Half detector:")
    circuit = bbcode_patch.generate_idling_circuit(1, half_detector=True)
    fig, ax = draw_qubit_gantt(
        bbcode_patch,
        qubit_ids=[
            bbcode_patch.ancilla_qubits[Basis.X][0].id,
            bbcode_patch.l_data_indices[0],
            bbcode_patch.r_data_indices[0],
            bbcode_patch.ancilla_qubits[Basis.Z][0].id,
        ],
        include_idle=True,
        label_entries=True,
        time_unit="us",
        time_scale="compressed",
        save_path="qubit_gantt.png",
        show=True,
    )
    circuit.detector_error_model()
    circuit.to_file("scattered-256.stim")
    exit(0)

    import sinter
    import pickle

    tasks = [
        sinter.Task(
            circuit=BivariateBicycleCodePatch(
                bbcode=bb,
                rounds=4,
                basis=Basis.Z,
                error_profile=NoiseModel(
                    before_clifford_error=DepolarizingNoise(p),
                    before_reset_error=DepolarizingNoise(p),
                    before_measurement_error=DepolarizingNoise(p),
                    gate_profile=GateProfile(),
                ),
            ).circuit(),
            json_metadata={"p": p},
        )
        for p in [0.01, 0.005]
    ]

    collected_stats = sinter.collect(
        tasks=tasks,
        decoders=["SinterBpOsdDecoder"],
        custom_decoders={"SinterBpOsdDecoder": SinterBpOsdDecoder()},
        num_workers=8,
        max_shots=100_000,
        max_errors=500,
    )

    print(collected_stats)
    pickle.dump(collected_stats, open("../collected_stats.pkl", "wb"))

    # patch = BivariateBicycleCodePatch(
    #     bbcode=bb,
    #     rounds=5,
    #     basis=Basis.Z,
    #     error_profile=NoiseModel(
    #         before_clifford_error=DepolarizingNoise(0.003),
    #         before_reset_error=DepolarizingNoise(0.003),
    #         before_measurement_error=DepolarizingNoise(0.003),
    #         gate_profile=GateProfile()
    #     )
    # )
    # circ = patch.circuit()
    exit(0)
