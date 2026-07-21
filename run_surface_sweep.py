#!/usr/bin/env python3
"""Generate surface-code memory-experiment circuits over a distance x noise
grid, decode them with sinter, and report logical error rates.

Mirrors the BivariateBicycleCode sweep in deformation.py's __main__, but for
SurfaceCodeModule: writes one noisy .stim file per (distance, p) combination
and immediately runs it through sinter.collect with pymatching.
"""

import argparse

import sinter
import stim

from deformation import SurfaceCodeModule, Basis
from error import NoiseModel, DepolarizingNoise, FlipNoise
from gate_profile import GateProfile


def build_circuit(distance: int, basis: Basis = Basis.X) -> stim.Circuit:
    module = SurfaceCodeModule(distance, basis)
    circ = module.initial_round()
    circ = module.qec_round(circ, detector_basis=[basis])

    repeated_block = stim.Circuit()
    repeated_block = module.qec_round(repeated_block, detector_basis=[Basis.X, Basis.Z])
    repeated_block = stim.CircuitRepeatBlock(module.d - 1, repeated_block)
    circ.append(repeated_block)

    circ = module.measure_all_data_qubits(circ)
    circ = module.finalize(circ)
    circ = module.observable(circ)
    return circ


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument(
        "--noise", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    parser.add_argument("--basis", type=str, choices=["X", "Z"], default="X")
    parser.add_argument("--max_shots", type=int, default=5000)
    parser.add_argument("--max_errors", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_csv", type=str, default="surface_sweep_results.csv")
    return parser


def main():
    args = build_parser().parse_args()
    basis = Basis[args.basis]

    tasks = []
    task_labels = []
    for distance in args.distances:
        circ = build_circuit(distance, basis)
        for p in args.noise:
            noise_profile = NoiseModel(
                gate_profile=GateProfile(),
                after_clifford_error=DepolarizingNoise(fixed_noise_level=p),
                before_measurement_error=FlipNoise(fixed_noise_level=p),
                after_reset_error=FlipNoise(fixed_noise_level=p),
            )
            noisy_circuit = noise_profile.apply(circ)

            filename = f"surface_d{distance}_p{p}.stim"
            noisy_circuit.to_file(filename)
            print(f"wrote {filename} "
                  f"(qubits={noisy_circuit.num_qubits}, "
                  f"detectors={noisy_circuit.num_detectors}, "
                  f"observables={noisy_circuit.num_observables})")

            tasks.append(
                sinter.Task(
                    circuit=noisy_circuit,
                    json_metadata={"d": distance, "p": p},
                )
            )
            task_labels.append((distance, p, filename))

    print(f"\nRunning {len(tasks)} tasks through sinter (pymatching)...")
    results = sinter.collect(
        num_workers=args.num_workers,
        tasks=tasks,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        decoders=["pymatching"],
        print_progress=True,
    )

    results.sort(key=lambda r: (r.json_metadata["d"], r.json_metadata["p"]))

    print("\n{:>4} {:>8} {:>10} {:>10} {:>14}".format(
        "d", "p", "shots", "errors", "logical_rate"
    ))
    with open(args.output_csv, "w") as f:
        f.write(sinter.CSV_HEADER + "\n")
        for r in results:
            rate = r.errors / r.shots if r.shots else float("nan")
            print("{:>4} {:>8} {:>10} {:>10} {:>14.5f}".format(
                r.json_metadata["d"], r.json_metadata["p"], r.shots, r.errors, rate
            ))
            f.write(r.to_csv_line() + "\n")

    print(f"\nResults written to {args.output_csv}")


if __name__ == "__main__":
    main()
