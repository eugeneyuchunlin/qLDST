import numpy as np
from typing import List, Optional, Tuple, Union
from typing_extensions import override
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import stim
import math

try:
    from basis import Basis
    from gate_profile import Gate, GateTaxonomy, GateProfile
    from circuit import MetaCircuit
except ImportError:
    from .basis import Basis
    from .gate_profile import Gate, GateTaxonomy, GateProfile
    from .circuit import MetaCircuit


class NoisePrimitive(ABC):

    @abstractmethod
    def get_instruction(
        self, gate_type: str, targets: List[int]
    ) -> stim.CircuitInstruction | None:
        pass


@dataclass
class DepolarizingNoise(NoisePrimitive):

    def __init__(self, fixed_noise_level: Optional[float] = None):
        self.fixed_noise_level = fixed_noise_level

    @override
    def get_instruction(
        self, gate_type: str, targets: List[int], noise_level=None
    ) -> stim.CircuitInstruction | None:
        if not GateTaxonomy.is_supported(gate_type):
            return None

        _gate_type = GateTaxonomy.identify_gate_type(gate_type)
        if _gate_type == "CLIFFORD_2Q":
            error_type = "DEPOLARIZE2"
        else:
            error_type = "DEPOLARIZE1"

        if noise_level is not None:
            return stim.CircuitInstruction(error_type, targets, [noise_level])
        elif self.fixed_noise_level is not None:
            return stim.CircuitInstruction(
                error_type, targets, [self.fixed_noise_level]
            )
        else:
            raise ValueError(
                "Either noise_level or fixed_noise_level in the __init__ must be provided."
            )


@dataclass
class IdlingNoise(NoisePrimitive):

    def __init__(
        self,
        basis: Basis,
        idle_multiplier: float,
        T1s: Optional[dict[int, float]] = None,
        T2s: Optional[dict[int, float]] = None,
        default_T1: float = 100 * 10 ** (-6),
        default_T2: float = 100 * 10 ** (-6),
    ):
        super().__init__()

        if T1s is not None:
            assert all(T1 > 0 for T1 in T1s.values()), "All T1 times must be positive."
            assert (
                max(T1s.values()) <= default_T1
            ), "Default T1 should be greater than or equal to the maximum T1 in the T1s dictionary."
        if T2s is not None:
            assert all(T2 > 0 for T2 in T2s.values()), "All T2 times must be positive."
            assert (
                max(T2s.values()) <= default_T2
            ), "Default T2 should be greater than or equal to the maximum T2 in the T2s dictionary."

        if T1s is not None and T2s is not None:
            assert min(T2s.values()) <= min(
                T1s.values()
            ), "Minimum T2 should be less than or equal to minimum T1 in the T1s and T2s dictionaries."
        assert (
            default_T2 <= default_T1
        ), "Default T2 should be less than or equal to default T1."

        self.T1s = T1s
        self.T2s = T2s

        self.default_T1 = default_T1
        self.default_T2 = default_T2

        self.idle_multiplier = idle_multiplier
        self.basis = basis

    def _resolve_param(
        self, q_id: Union[int, List[int]], param_map: dict | None, default_val: float
    ) -> Union[float, np.ndarray]:
        """
        Helper to handle both scalar integers and lists/arrays of integers.
        """
        if hasattr(q_id, "__iter__") and not isinstance(q_id, (int, float)):
            return np.array(
                [self._resolve_param(q, param_map, default_val) for q in q_id]
            )

        if param_map is not None and q_id in param_map:
            return param_map[q_id]
        return default_val

    def _get_T1(self, q_id: Union[int, List[int]]) -> Union[float, np.ndarray]:
        return self._resolve_param(q_id, self.T1s, self.default_T1)

    def _get_T2(self, q_id: Union[int, List[int]]) -> Union[float, np.ndarray]:
        return self._resolve_param(q_id, self.T2s, self.default_T2)

    @override
    def get_instruction(
        self, time: float, q_id: int
    ) -> stim.CircuitInstruction | None:  # time in seconds

        px = (1 - np.exp(-time / self._get_T1(q_id))) / 4
        py = px
        pz = (1 - np.exp(-time / self._get_T2(q_id))) / 2 - px

        # print(f"time: {time}, px: {px}, py: {py}, pz: {pz}")
        assert (
            px.min() >= 0 and py.min() >= 0 and pz.min() >= 0
        ), "Error probabilities must be non-negative. Consider changing the T1/T2 times"

        def _error(p1, p2):
            return float(
                1 - (1 - p1 * self.idle_multiplier) * (1 - p2 * self.idle_multiplier)
            )

        # if self.basis.value == Basis.Z.value:
        #     err = _error(px, py)
        #     error = stim.CircuitInstruction('X_ERROR', [q_id], [_error(px, py)])
        # elif self.basis.value == Basis.X.value:
        #     error = stim.CircuitInstruction('Z_ERROR', [q_id], [_error(pz, py)])
        # else:
        #     raise ValueError("Invalid basis")

        error = stim.CircuitInstruction(
            "PAULI_CHANNEL_1", [q_id], [px, py, pz], tag=f"time={time}s"
        )

        return error


class FlipNoise(NoisePrimitive):

    def __init__(
        self, fixed_noise_level: Optional[float] = None, basis: Basis = Basis.Z
    ):
        self.fixed_noise_level = fixed_noise_level
        self.basis = basis

    @override
    def get_instruction(
        self, gate_type: str, targets: List[int], noise_level: float = None
    ) -> stim.CircuitInstruction | None:
        error_type = None

        if not (
            GateTaxonomy.is_measurement(gate_type) or GateTaxonomy.is_reset(gate_type)
        ):
            return None

        _basis = GateTaxonomy.get_meas_reset_type(gate_type)

        if _basis == Basis.Z:
            error_type = "X_ERROR"
        elif _basis == Basis.X:
            error_type = "Z_ERROR"
        else:
            return None

        if noise_level is not None:
            return stim.CircuitInstruction(error_type, targets, [noise_level])
        elif self.fixed_noise_level is not None:
            return stim.CircuitInstruction(
                error_type, targets, [self.fixed_noise_level]
            )
        else:
            raise ValueError(
                "Either noise_level or fixed_noise_level in the __init__ must be provided."
            )


class NoiseModel:

    def __init__(
        self,
        gate_profile: GateProfile,
        *,
        after_clifford_error: Optional[list[NoisePrimitive] | NoisePrimitive] = None,
        before_clifford_error: Optional[list[NoisePrimitive] | NoisePrimitive] = None,
        before_measurement_error: Optional[
            list[NoisePrimitive] | NoisePrimitive
        ] = None,
        after_measurement_error: Optional[list[NoisePrimitive] | NoisePrimitive] = None,
        before_reset_error: Optional[list[NoisePrimitive] | NoisePrimitive] = None,
        after_reset_error: Optional[list[NoisePrimitive] | NoisePrimitive] = None,
        after_each_round: Optional[list[NoisePrimitive] | NoisePrimitive] = None,
    ):
        self.gate_profile = gate_profile
        self.after_clifford_error = after_clifford_error
        self.before_clifford_error = before_clifford_error
        self.before_measurement_error = before_measurement_error
        self.after_measurement_error = after_measurement_error
        self.before_reset_error = before_reset_error
        self.after_reset_error = after_reset_error

    def _noisy_instructions(self, instructions):
        """Yields each instruction with per-gate noise interleaved, for a
        sequence of non-REPEAT instructions (each either a
        ``stim.CircuitInstruction`` or a ``MetaInstruction``). Every yielded
        instruction -- the original gate included -- is an already-built
        object, never decomposed and reconstructed, since
        ``generate_noise()``'s results are already ``stim.CircuitInstruction``
        instances and the original gate doesn't need to change at all."""
        for instruction in instructions:
            gate_type = instruction.name
            if gate_type in GateTaxonomy.ALL_GATES:
                targets = list(instruction.targets_copy())
                pre_gate_noise, post_gate_noise = self.generate_noise(
                    gate_type, targets
                )
                yield from pre_gate_noise
                yield instruction
                yield from post_gate_noise
            else:
                yield instruction

    def _materialize_entries(self, instructions) -> stim.Circuit:
        """Builds a real ``stim.Circuit`` chunk from a flat run of
        instructions with noise applied, via ``MetaCircuit`` (one bulk
        parse) instead of one ``stim.Circuit.append()`` per instruction."""
        chunk = MetaCircuit()
        for instr in self._noisy_instructions(instructions):
            chunk.append_instruction(instr)
        return chunk.to_stim()

    def apply(self, circuit: stim.Circuit | MetaCircuit) -> stim.Circuit | MetaCircuit:
        """
        Applies this noise model to ``circuit``, returning the same kind
        (``stim.Circuit`` or ``MetaCircuit``) back.

        Accumulates into a ``MetaCircuit`` internally and only materializes
        a real ``stim.Circuit`` in bulk, instead of one
        ``stim.Circuit.append()`` call per instruction/noise event --
        ``stim.Circuit.append()`` costs ~100us/call, mostly from crossing
        into the compiled extension on every single instruction, and this
        model appends noise around every gate.

        A ``REPEAT`` block (which ``MetaCircuit`` can't represent
        compactly) only ever appears in ``stim.Circuit`` input; when one is
        found, the flat instructions around it are still fast-pathed via
        ``MetaCircuit`` and concatenated with the block itself (recursively
        noised) via ``stim.Circuit``'s bulk ``+=``.
        """
        if isinstance(circuit, MetaCircuit):
            out = MetaCircuit()
            for instr in self._noisy_instructions(circuit):
                out.append_instruction(instr)
            return out

        new_circuit = stim.Circuit()
        pending: list[stim.CircuitInstruction] = []
        for instruction in circuit:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                if pending:
                    new_circuit += self._materialize_entries(pending)
                    pending = []
                noisy_body = self.apply(instruction.body_copy())
                new_circuit.append(
                    stim.CircuitRepeatBlock(instruction.repeat_count, noisy_body)
                )
                continue
            pending.append(instruction)

        if pending:
            new_circuit += self._materialize_entries(pending)

        return new_circuit

    def generate_noise(
        self, gate_type: str, targets: list[int]
    ) -> NoisePrimitive | None:
        pre_gate_noise = []
        post_gate_noise = []

        def _apply(error_models, dest_list):

            if error_models is not None:
                if isinstance(error_models, NoisePrimitive):
                    error_models = [error_models]

                for error_model in error_models:
                    # use either gate error rate or model's error rate
                    error_rate = (
                        self.gate_profile[gate_type].error_rate
                        if error_model.fixed_noise_level is None
                        else error_model.fixed_noise_level
                    )
                    noise = error_model.get_instruction(gate_type, targets, error_rate)
                    if noise is not None:
                        dest_list.append(noise)

        def _apply_reset(error_models, dest_list):
            # The reason why using _apply_reset because _apply will use the measurement error specified by M
            if error_models is not None:
                if isinstance(error_models, NoisePrimitive):
                    error_models = [error_models]

                for error_model in error_models:
                    error_rate = (
                        self.gate_profile["R"].error_rate
                        if error_model.fixed_noise_level is None
                        else error_model.fixed_noise_level
                    )
                    noise = error_model.get_instruction(gate_type, targets, error_rate)
                    if noise is not None:
                        dest_list.append(noise)

        if GateTaxonomy.is_clifford(gate_type):
            _apply(self.before_clifford_error, pre_gate_noise)
            _apply(self.after_clifford_error, post_gate_noise)

        if GateTaxonomy.is_measurement(gate_type):
            _apply(self.before_measurement_error, pre_gate_noise)
            _apply(self.after_measurement_error, post_gate_noise)

        if GateTaxonomy.is_reset(gate_type):
            _apply_reset(self.before_reset_error, pre_gate_noise)
            _apply_reset(self.after_reset_error, post_gate_noise)

        # else:
        #     raise ValueError(f"Unknown gate type: {gate_type}")

        return pre_gate_noise, post_gate_noise

    # def generate_idling_noise(self, time: float, q_id: int) -> Optional[str]:
    #     if self.idling_error is not None:
    #         return self.idling_error.get_instruction(time, q_id)
    #     else:
    #         return None


class SI1000NoiseProfile(NoiseModel):

    def __init__(self, noise_level: float, gate_profile: GateProfile):
        self.noise_level = noise_level

        super().__init__(
            gate_profile=gate_profile,
            before_clifford_error=DepolarizingNoise(),
            before_measurement_error=DepolarizingNoise(),
            before_reset_error=FlipNoise(),
        )

    def generate_noise(
        self, gate_type: str, targets: list[int]
    ) -> NoisePrimitive | None:

        return super().generate_noise(gate_type, targets)


class IdlingRecord:

    def __init__(self, q_id: int):
        self.q_id = q_id
        self.idling_time = 0.0
        self.missed_gates: list[Gate] = []

    def record_missed_gate(self, gate: Gate):
        self.missed_gates.append(gate)
        self.idling_time += gate.duration

    def record_idle_time(self, time: float):
        self.idling_time += time

    def reset(self):
        self.idling_time = 0.0
        self.missed_gates.clear()


# ---------------------------------------------------------------------------
# Power-dilution scaling law for Rydberg two-qubit gate error.
#
# Intrinsic model (two-term, universal form around the optimum):
#     E_int(x) = E_min * ( (2/3) / x + (1/3) * x**2 ),   x = Omega / Omega_opt
#
# Dilution law:  Omega(r) = Omega_0 * r**(-alpha)
#
# Two-channel model (recommended for measured gate errors):
#     E_total = E_tech + E_doppler + E_int(x)
# where E_tech is dilution-insensitive technical noise (laser phase noise,
# pulse-envelope imperfections, calibration drift) and E_doppler is the
# Doppler-detuning error, which scales as 1/Omega^2 [Ma et al. 2023,
# arXiv:2305.05493]. For a 5e-3-class Rb two-photon gate, error budgets
# [Evered et al., Nature 622, 268 (2023); Ma et al. 2023] suggest
# E_tech ~ 1e-3 to 3e-3 (nominal 2e-3).
#
# undo_dilution(gate_error, r): given the error at array-size ratio r
# (relative to baseline), return the error obtained by undoing the
# dilution, i.e. concentrating the same total power on N/r sites.
# ---------------------------------------------------------------------------

E_MIN_DEFAULT = 9.3e-4  # 87Rb n=100, tau=340us, B/2pi=83 MHz (Saffman Eq. 26)
E_TECH_NOMINAL = 2e-3  # nominal technical floor for a 5e-3-class Rb gate


def _solve_x(gate_error, e_min, branch="decay"):
    """Solve E_int(x) = gate_error for x = Omega/Omega_opt.

    Multiplying E/E_min = (2/3)/x + (1/3)x^2 by 3x gives the depressed cubic
        x^3 - 3*e*x + 2 = 0,   e = gate_error / e_min  (e >= 1).
    Roots: one negative (unphysical) and two positive:
        decay branch    0 < x <= 1  (Omega below optimum)
        blockade branch x >= 1      (Omega above optimum)
    """
    e = gate_error / e_min
    if e < 1.0 - 1e-12:
        raise ValueError(
            f"intrinsic error {gate_error:.3e} is below the intrinsic floor "
            f"E_min = {e_min:.3e}; no operating point exists."
        )
    e = max(e, 1.0)

    # Trigonometric solution of x^3 + p*x + q = 0 with p = -3e, q = 2.
    p, q = -3.0 * e, 2.0
    m = 2.0 * math.sqrt(-p / 3.0)  # = 2*sqrt(e)
    arg = max(-1.0, min(1.0, 3.0 * q / (p * m)))  # = -1/e^{3/2}
    phi = math.acos(arg)
    roots = sorted(m * math.cos((phi - 2.0 * math.pi * k) / 3.0) for k in range(3))
    # roots[0] < 0 (unphysical), roots[1] = decay branch, roots[2] = blockade
    if branch == "decay":
        return roots[1]
    elif branch == "blockade":
        return roots[2]
    raise ValueError("branch must be 'decay' or 'blockade'")


def gate_error_at(x, e_min=E_MIN_DEFAULT):
    """Universal intrinsic error curve E_int(x) = E_min*((2/3)/x + (1/3)x^2)."""
    return e_min * ((2.0 / 3.0) / x + (x * x) / 3.0)


def undo_dilution(
    gate_error,
    r,
    alpha=0.5,
    e_min=E_MIN_DEFAULT,
    branch="decay",
    clamp_at_opt=True,
    verbose=False,
):
    """
    Intrinsic-only inversion (optimistic bound: attributes the entire
    gate_error to the decay/blockade tradeoff, so power concentration
    scales all of it). Prefer undo_dilution_two_channel for measured errors.

    Parameters
    ----------
    gate_error : float
        Intrinsic gate error at array-size ratio r. Must be >= e_min.
    r : float
        Qubit-count ratio N / N_0. Undoing concentrates power by r**alpha.
    alpha : float
        Dilution exponent: 0.5 for single-photon or two-photon with one
        power-limited beam (realistic Rb case); 1.0 for two-photon with
        both beams power-limited (pessimistic).
    e_min : float
        Intrinsic error floor for the hardware parameter set.
    branch : str
        'decay' (x < 1, the branch dilution puts you on) or 'blockade'.
    clamp_at_opt : bool
        If True, cap the improvement at the optimum (return e_min) rather
        than climbing the blockade branch when x would exceed 1.
    """
    if r <= 0:
        raise ValueError("r must be positive")

    x_r = _solve_x(gate_error, e_min, branch=branch)
    x_0 = x_r * r**alpha

    clamped = clamp_at_opt and x_0 > 1.0
    x_eval = 1.0 if clamped else x_0
    e_0 = gate_error_at(x_eval, e_min)

    if verbose:
        print(
            f"  E(r)={gate_error:.3e} at r={r}, alpha={alpha}: "
            f"x_r={x_r:.4f} -> x_0={x_0:.4f}"
            + (" (clamped at optimum)" if clamped else "")
        )
        print(f"  E(1) = {e_0:.3e}  (improvement x{gate_error / e_0:.2f})")
    return e_0


def undo_dilution_two_channel(
    gate_error,
    r,
    alpha=0.5,
    e_min=E_MIN_DEFAULT,
    e_tech=E_TECH_NOMINAL,
    e_doppler=0.0,
    branch="decay",
    clamp_at_opt=True,
    verbose=False,
):
    """
    Two-channel inversion:  E_total = E_tech + E_doppler + E_int.

    Only the intrinsic part follows the universal curve; E_tech is held
    fixed (conservative), and E_doppler scales as 1/Omega^2, i.e. improves
    by r**(2*alpha) under an r-fold concentration [Ma et al. 2023].

    Parameters (beyond undo_dilution's)
    ----------
    gate_error : float
        Total measured gate error at array-size ratio r.
    e_tech : float
        Dilution-insensitive technical error at ratio r (phase noise,
        pulse envelope, calibration). Nominal 2e-3 for a 5e-3-class Rb
        gate; sensitivity range 1e-3 to 3e-3.
    e_doppler : float
        Doppler-detuning error at ratio r, scaled separately (~1e-3 in
        the Rb budget if you split it out; fold it into e_tech to stay
        conservative).

    Returns
    -------
    float : total gate error at baseline (r -> 1).
    """
    e_int_r = gate_error - e_tech - e_doppler
    if e_int_r < e_min - 1e-12:
        raise ValueError(
            f"residual intrinsic error {e_int_r:.3e} (= gate_error - e_tech "
            f"- e_doppler) is below E_min = {e_min:.3e}; reduce e_tech / "
            f"e_doppler or check the budget split."
        )

    e_int_0 = undo_dilution(
        e_int_r, r, alpha=alpha, e_min=e_min, branch=branch, clamp_at_opt=clamp_at_opt
    )
    e_dop_0 = e_doppler * r ** (-2.0 * alpha)
    e_total = e_tech + e_dop_0 + e_int_0

    if verbose:
        print(
            f"  budget at r={r}: tech={e_tech:.2e} (fixed), "
            f"doppler={e_doppler:.2e} -> {e_dop_0:.2e}, "
            f"intrinsic={e_int_r:.2e} -> {e_int_0:.2e}"
        )
        print(
            f"  E_total(1) = {e_total:.3e}  "
            f"(improvement x{gate_error / e_total:.2f})"
        )
    return e_total


def dilution_table(
    baseline_error, ratios=(2, 4, 8), alphas=(0.5, 1.0), two_channel=True, **kwargs
):
    """
    Utility-case table: given the gate error at array size N, predict the
    error at N/m for each m in `ratios` (i.e. undo an m-fold dilution).
    Returns {alpha: {m: error}} and prints a formatted table.
    """
    fn = undo_dilution_two_channel if two_channel else undo_dilution
    out = {}
    header = "  N/m   " + "".join(f"alpha={a:<11}" for a in alphas)
    print(header)
    for m in (1,) + tuple(ratios):
        row = f"  N/{m:<4}"
        for a in alphas:
            e = baseline_error if m == 1 else fn(baseline_error, m, alpha=a, **kwargs)
            out.setdefault(a, {})[m] = e
            row += f"{e:<17.3e}"
        print(row)
    return out


def scaled_dilution(
    baseline_error,
    r,
    alpha=1,
    f_tech=0.5,
    x0=0.2,
    f_doppler=0.0,
    clamp_at_opt=True,
    verbose=False,
):
    """
    Fractional-budget scaling law: floor-safe for sweeps over baseline_error.

    Treats each baseline_error as a self-consistent hypothetical machine
    whose error budget has fixed *proportions* rather than fixed absolute
    floors: a fraction f_tech is dilution-insensitive technical noise, a
    fraction f_doppler scales as 1/Omega^2, and the remainder is intrinsic
    (decay + blockade) sitting at position x0 on its own universal curve.
    The machine's own E_min is implied by (baseline_error, f_tech, x0) and
    is never violated -- any positive baseline_error is a valid input.

    Parameters
    ----------
    baseline_error : float
        Total gate error of the (hypothetical) baseline machine.
    r : float
        Qubit-count ratio N/N_0 (r > 1 dilutes, r < 1 concentrates:
        e.g. r = 1/8 for the N/8 utility case).
    f_tech : float
        Technical fraction of the baseline budget (~0.4-0.6 from the
        error budgets of Evered et al. 2023 / Ma et al. 2023).
    x0 : float
        Baseline position of the intrinsic part on its universal curve
        (x = Omega/Omega_opt). x0 < 1 = power-starved baseline with
        improvement headroom under concentration (default 0.2, the
        operating point inferred for a 5e-3-class intrinsic error).
        x0 = 1 = baseline at its own optimum: conservative for dilution
        (r > 1) but leaves ZERO improvement from concentration (r < 1).
    f_doppler : float
        Doppler fraction, scaled separately as r**(-2*alpha); fold into
        f_tech to stay conservative.
    clamp_at_opt : bool
        If True, concentration never overshoots the optimum: excess power
        is attenuated and the intrinsic part saturates at its floor.
    """
    if baseline_error <= 0 or r <= 0:
        raise ValueError("baseline_error and r must be positive")
    f_int = 1.0 - f_tech - f_doppler
    if f_int < 0:
        raise ValueError("f_tech + f_doppler must not exceed 1")

    def curve(x):  # E_int(x) / E_min
        return (2.0 / 3.0) / x + (x * x) / 3.0

    x_r = x0 * r ** (-alpha)
    clamped = clamp_at_opt and x_r > 1.0 >= x0
    x_eval = 1.0 if clamped else x_r
    intrinsic_ratio = curve(x_eval) / curve(x0)  # E_int(r) / E_int(1)

    ratio = f_tech + f_doppler * r ** (-2.0 * alpha) + f_int * intrinsic_ratio
    e = baseline_error * ratio
    if verbose:
        print(
            f"  p={baseline_error:.3e}, r={r}, alpha={alpha}, "
            f"f_tech={f_tech}, x0={x0}: x -> {x_r:.3f}"
            + (" (clamped at optimum)" if clamped else "")
            + f", E(r) = {e:.3e} (ratio {ratio:.3f})"
        )
    return e


def sweep_table(
    baseline_errors=(1e-3, 2e-3, 3e-3),
    ratios=(2, 4, 8),
    alpha=0.5,
    f_tech=0.5,
    x0=0.2,
    **kwargs,
):
    """Sweep-safe utility table: error at N/m for each swept baseline p_N."""
    print(f"  alpha={alpha}, f_tech={f_tech}, x0={x0}")
    print("  p_N      " + "".join(f"N/{m:<12}" for m in ratios))
    out = {}
    for p in baseline_errors:
        row = f"  {p:<9.1e}"
        for m in ratios:
            e = scaled_dilution(p, 1.0 / m, alpha=alpha, f_tech=f_tech, x0=x0, **kwargs)
            out.setdefault(p, {})[m] = e
            row += f"{e:<14.3e}"
        print(row)
    return out


if __name__ == "__main__":

    circuit = stim.Circuit.from_file("./bb.stim")
    new_circuit = NoiseModel(
        gate_profile=GateProfile(),
        after_clifford_error=DepolarizingNoise(fixed_noise_level=0.001),
        before_measurement_error=DepolarizingNoise(fixed_noise_level=0.001),
        after_reset_error=FlipNoise(fixed_noise_level=0.001, basis=Basis.X),
    ).apply(circuit)

    new_circuit.to_file("./bb_with_noise.stim")

    from utils.helper import view_circuit_timeline_diagram

    view_circuit_timeline_diagram(new_circuit)
