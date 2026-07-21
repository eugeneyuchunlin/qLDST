#!/usr/bin/env python3
"""Build a code-deformation (BB-code -> surface-code teleportation) circuit
via Deformation, apply noise, write it to a .stim file, and decode it with
Relay-BP (reusing runner.py's tuned SinterDecoder_RelayBP setup) to verify
the protocol degrades gracefully with noise instead of just checking the
noiseless case.
"""

import argparse

import sinter
import stim

from deformation import BivariateBicycleCode, Deformation
from error import NoiseModel, DepolarizingNoise, FlipNoise
from gate_profile import GateProfile
from runner import build_relay_bp_arguments


def build_circuit(bb_distance: int, surface_distance: int, bb_rounds: int, surface_rounds: int, bridge_copies: int) -> stim.Circuit:
    code = BivariateBicycleCode.generate_code(distance=bb_distance)
    deformation = Deformation(code, surface_distance, logical_index=0)
    return deformation.build_circuit(
        bb_rounds=bb_rounds,
        surface_rounds=surface_rounds,
        bridge_copies=bridge_copies,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bb-distance", type=int, default=12,
                         help="BB (source/memory) code distance. Must have a tuned "
                              "Relay-BP memory strength in runner.RELAY_MEMORY_STRENGTHS (12 or 18).")
    parser.add_argument("--surface-distance", type=int, default=5,
                         help="Surface code (destination/compute) distance.")
    parser.add_argument("--bb-rounds", type=int, default=None,
                         help="QEC rounds run on the source before the transfer. Defaults to bb-distance.")
    parser.add_argument("--surface-rounds", type=int, default=None,
                         help="QEC rounds run on the destination after the transfer. Defaults to surface-distance.")
    parser.add_argument("--bridge-copies", type=int, default=3,
                         help="Redundant parallel ancillas for the one-shot joint Z_A Z_B "
                              "measurement (repetition-code-style protection against a single "
                              "bad ancilla measurement -- see Deformation docstring).")
    parser.add_argument("--noise", type=float, nargs="+", default=[1e-3, 2e-3, 5e-3])
    parser.add_argument("--max_shots", type=int, default=10_000)
    parser.add_argument("--max_errors", type=int, default=2_000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_csv", type=str, default="deformation_results.csv")
    return parser


def main():
    args = build_parser().parse_args()
    bb_rounds = args.bb_rounds or args.bb_distance
    surface_rounds = args.surface_rounds or args.surface_distance

    circ = build_circuit(
        args.bb_distance, args.surface_distance, bb_rounds, surface_rounds, args.bridge_copies
    )
    print(f"circuit: qubits={circ.num_qubits} detectors={circ.num_detectors} observables={circ.num_observables}")

    decoder_arguments = build_relay_bp_arguments(args.bb_distance)

    tasks = []
    for p in args.noise:
        noise_profile = NoiseModel(
            gate_profile=GateProfile(),
            after_clifford_error=DepolarizingNoise(fixed_noise_level=p),
            before_measurement_error=FlipNoise(fixed_noise_level=p),
            after_reset_error=FlipNoise(fixed_noise_level=p),
        )
        noisy_circuit = noise_profile.apply(circ)

        filename = f"deformation_bb{args.bb_distance}_surf{args.surface_distance}_p{p}.stim"
        noisy_circuit.to_file(filename)
        print(f"wrote {filename}")

        tasks.append(
            sinter.Task(
                circuit=noisy_circuit,
                json_metadata={
                    "bb_distance": args.bb_distance,
                    "surface_distance": args.surface_distance,
                    "p": p,
                },
                decoder="relay-bp",
            )
        )

    print(f"\nRunning {len(tasks)} tasks through sinter (relay-bp)...")
    results = sinter.collect(
        num_workers=args.num_workers,
        tasks=tasks,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        print_progress=True,
        **decoder_arguments,
    )

    results.sort(key=lambda r: r.json_metadata["p"])

    print("\n{:>8} {:>10} {:>10} {:>14}".format("p", "shots", "errors", "logical_rate"))
    with open(args.output_csv, "w") as f:
        f.write(sinter.CSV_HEADER + "\n")
        for r in results:
            rate = r.errors / r.shots if r.shots else float("nan")
            print("{:>8} {:>10} {:>10} {:>14.5f}".format(r.json_metadata["p"], r.shots, r.errors, rate))
            f.write(r.to_csv_line() + "\n")

    print(f"\nResults written to {args.output_csv}")


if __name__ == "__main__":
    main()
