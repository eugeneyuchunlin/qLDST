import enum
import re
from dataclasses import dataclass, field
from typing import Union

try:
    from basis import Basis
except ImportError:
    from .basis import Basis


@dataclass(kw_only=True)
class Gate:
    name: str
    duration: float | None = None
    error_rate: float | None = None


@dataclass(kw_only=True)
class Gate2Q(Gate):
    control: int
    target: int


class GateTaxonomy:
    CLIFFORD_1Q = {"H", "S", "X", "Y", "Z"}
    CLIFFORD_2Q = {"CNOT", "CZ", "CX", "ZCX", "ZCZ"}
    MEASUREMENT = {"MR", "MRX", "M", "MX", "MZ", "MRZ"}
    RESET = {"R", "RX", "MR", "MRX"}

    CLIFFORD = CLIFFORD_1Q.union(CLIFFORD_2Q)
    ALL_GATES = CLIFFORD_1Q.union(CLIFFORD_2Q, MEASUREMENT, RESET)

    @staticmethod
    def _resolve_name(gate: Union[str, Gate]) -> str:
        """Extracts the string name from either a string or a Gate object."""
        if isinstance(gate, Gate):
            return gate.name
        return gate

    @staticmethod
    def is_clifford(gate_type: str | Gate) -> bool:
        gate_type = GateTaxonomy._resolve_name(gate_type)
        return (
            gate_type in GateTaxonomy.CLIFFORD_1Q
            or gate_type in GateTaxonomy.CLIFFORD_2Q
        )

    @staticmethod
    def is_clifford_1q(gate_type: str | Gate) -> bool:
        gate_type = GateTaxonomy._resolve_name(gate_type)
        return gate_type in GateTaxonomy.CLIFFORD_1Q

    @staticmethod
    def is_clifford_2q(gate_type: str | Gate) -> bool:
        gate_type = GateTaxonomy._resolve_name(gate_type)
        return gate_type in GateTaxonomy.CLIFFORD_2Q

    @staticmethod
    def is_cnot(gate_type: str | Gate) -> bool:
        gate_type = GateTaxonomy._resolve_name(gate_type)
        return gate_type in {"CNOT", "CX", "ZCX"}

    @staticmethod
    def is_cz(gate_type: str | Gate) -> bool:
        gate_type = GateTaxonomy._resolve_name(gate_type)
        return gate_type in {"CZ", "ZCZ"}

    @staticmethod
    def is_measurement(gate_type: str | Gate) -> bool:
        gate_type = GateTaxonomy._resolve_name(gate_type)
        return gate_type in GateTaxonomy.MEASUREMENT

    @staticmethod
    def is_reset(gate_type: str | Gate) -> bool:
        gate_type = GateTaxonomy._resolve_name(gate_type)
        return gate_type in GateTaxonomy.RESET

    @staticmethod
    def is_supported(gate_type: str | Gate) -> bool:
        gate_type = GateTaxonomy._resolve_name(gate_type)
        return gate_type in GateTaxonomy.ALL_GATES

    @staticmethod
    def get_meas_reset_type(gate_type: str | Gate) -> str:
        gate_type = GateTaxonomy._resolve_name(gate_type)
        if gate_type in {"M", "MR", "MRZ", "MZ", "RZ", "R"}:
            return Basis.Z
        elif gate_type in {"MX", "MRX", "RX"}:
            return Basis.X
        else:
            raise ValueError(f"Unsupported measurement gate type: {gate_type}")

    @staticmethod
    def identify_gate_type(gate_type: str | Gate) -> str:
        gate_type = GateTaxonomy._resolve_name(gate_type)
        if gate_type in GateTaxonomy.CLIFFORD_1Q:
            return "CLIFFORD_1Q"
        elif gate_type in GateTaxonomy.CLIFFORD_2Q:
            return "CLIFFORD_2Q"
        elif gate_type in GateTaxonomy.MEASUREMENT:
            return "MEASUREMENT"
        elif gate_type in GateTaxonomy.RESET:
            return "RESET"
        else:
            raise ValueError(f"Unsupported gate type: {gate_type}")


@dataclass
class GateProfile:
    gates: dict[str, Gate] = field(default_factory=dict)
    DEFAULTS = {
        "CNOT": Gate2Q(
            name="CNOT",
            duration=0.1 * 10 ** (-6),
            error_rate=0.001,
            control=None,
            target=None,
        ),
        "CX": Gate2Q(
            name="CX",
            duration=0.1 * 10 ** (-6),
            error_rate=0.001,
            control=None,
            target=None,
        ),
        "CZ": Gate2Q(
            name="CZ",
            duration=0.1 * 10 ** (-6),
            error_rate=0.001,
            control=None,
            target=None,
        ),
        "H": Gate(name="H", duration=0.1 * 10 ** (-6), error_rate=0.0001),
        "MR": Gate(name="MR", duration=100 * 10 ** (-3), error_rate=0.001),
        "M": Gate(name="M", duration=100 * 10 ** (-3), error_rate=0.001),
        "R": Gate(name="R", duration=100 * 10 ** (-3), error_rate=0.001),
        "MX": Gate(name="MX", duration=100 * 10 ** (-3), error_rate=0.001),
        "RX": Gate(name="RX", duration=100 * 10 ** (-3), error_rate=0.001),
    }

    # CNOT and CX are literal stim synonyms for the same instruction, and CZ
    # is the same entangling gate viewed through gate-set transpilation (see
    # gate_set_transpiler.py's CX<->CZ transpilers) -- so a *real* gate
    # profile needs all three to share duration/error_rate, or noise/timing
    # would silently depend on which gate-set representation a circuit
    # happens to be in. Not enforced unconditionally in __init__ since tests
    # elsewhere construct GateProfiles with deliberately distinct per-gate
    # dummy values to isolate gate-name lookups; call sync_two_qubit_gates
    # explicitly wherever a profile is meant to model real hardware (e.g.
    # from_string, sim.py's --gate_profile parsing).
    TWO_QUBIT_SYNC_GATES = ("CNOT", "CX", "CZ")

    def __init__(self, **kwargs):
        self.gates = self.DEFAULTS.copy()
        if "gates" in kwargs and isinstance(kwargs["gates"], dict):
            self.gates.update(kwargs.pop("gates"))

        for gate_name, gate_obj in kwargs.items():
            if isinstance(gate_obj, Gate):
                self.gates[gate_name] = gate_obj

    @classmethod
    def sync_two_qubit_gates(cls, gates: dict[str, Gate]) -> dict[str, Gate]:
        """Return a copy of ``gates`` with CNOT/CX/CZ synced (see
        TWO_QUBIT_SYNC_GATES).

        If more than one of them is present in ``gates`` and they disagree,
        that's a real misconfiguration -- raise rather than silently pick
        one. Otherwise, propagate whichever one is present onto the others
        that are absent (those will otherwise fall back to independent,
        possibly-mismatched GateProfile.DEFAULTS entries).
        """
        gates = dict(gates)
        present = [name for name in cls.TWO_QUBIT_SYNC_GATES if name in gates]
        if not present:
            return gates

        source_name = present[0]
        source = gates[source_name]
        for name in present[1:]:
            other = gates[name]
            if (other.duration, other.error_rate) != (source.duration, source.error_rate):
                raise ValueError(
                    f"Conflicting two-qubit gate configuration: '{source_name}' "
                    f"(duration={source.duration}, error_rate={source.error_rate}) vs "
                    f"'{name}' (duration={other.duration}, error_rate={other.error_rate}). "
                    "CNOT/CX/CZ represent the same physical two-qubit gate and must "
                    "share duration/error_rate."
                )

        for name in cls.TWO_QUBIT_SYNC_GATES:
            if name in present:
                continue
            default = cls.DEFAULTS.get(name)
            if isinstance(default, Gate2Q):
                gates[name] = Gate2Q(
                    name=name,
                    duration=source.duration,
                    error_rate=source.error_rate,
                    control=default.control,
                    target=default.target,
                )
            else:
                gates[name] = Gate(
                    name=name, duration=source.duration, error_rate=source.error_rate
                )
        return gates

    def get_gate(self, gate_name: str) -> Gate:
        try:
            return self.gates[gate_name]
        except KeyError:
            raise ValueError(f"Gate '{gate_name}' not found in gate profile.")

    def __getitem__(self, item):
        return self.get_gate(item)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{';'.join(f'{name},{gate.duration:.2e},{gate.error_rate:.2e}' for name, gate in self.gates.items())}"

    @classmethod
    def from_string(cls, profile_str: str):
        """
        Parses a string like 'GateProfile(CNOT: 1.0e-07s, 0.001, ...)'
        into a GateProfile object.
        """

        pattern = r"([A-Z]+),\s*([\d\.eE+-]+),\s*([\d\.eE+-]+)"

        matches = re.findall(pattern, profile_str)

        gates_dict = {}
        for name, duration_str, error_rate_str in matches:
            gates_dict[name] = Gate(
                name=name,
                duration=float(duration_str),
                error_rate=float(error_rate_str),
            )

        return cls(gates=cls.sync_two_qubit_gates(gates_dict))


if __name__ == "__main__":

    gate_profile = GateProfile.from_string(
        "CNOT,1.78e-05,1.00e-03;CZ,1.00e-07,1.00e-03;H,1.00e-07,1.00e-04;MR,1.99e-06,1.00e-03;M,1.00e-03,1.00e-03;R,1.00e-01,1.00e-03"
    )
    print(gate_profile.gates["CNOT"])
    print(gate_profile.gates["MR"])
    print(gate_profile.__str__())
