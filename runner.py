from typing import List, Optional

import argparse
import json
import os
import pickle

import sinter
import stim
from ldpc import SinterBpOsdDecoder as LDPCSinterBpOsdDecoder
from qldpc import decoders
from relay_bp.stim import SinterDecoder_RelayBP

RELAY_MEMORY_STRENGTHS = {
    18: (-0.161, 0.815),
    12: (-0.22628432386414646, 0.6216020925981884),
}


def run_simulations(
    *,
    circuit: str,
    json_metadata: dict,
    num_workers: int,
    max_shots: int,
    max_errors: int,
    save_resume_filepath: Optional[str] = None,
    **kwargs,
):
    circ = stim.Circuit.from_file(circuit)
    decoder = None
    if "custom_decoders" in kwargs:
        if "relay-bp" in kwargs["custom_decoders"]:
            decoder = "relay-bp"
        elif "sliding-window-relay-bp" in kwargs["custom_decoders"]:
            decoder = "sliding-window-relay-bp"
        elif "bposd" in kwargs["custom_decoders"]:
            decoder = "bposd"

    task = sinter.Task(
        circuit=circ,
        json_metadata=json_metadata,
        decoder=decoder,
    )
    collected_stats: List[sinter.TaskStats] = sinter.collect(
        num_workers=num_workers,
        tasks=[task],
        max_shots=max_shots,
        max_errors=max_errors,
        save_resume_filepath=save_resume_filepath,
        print_progress=True,
        **kwargs,
    )
    return collected_stats


def get_distance(args: argparse.Namespace, json_metadata: dict) -> int:
    distance = (
        args.distance if args.distance is not None else json_metadata.get("distance")
    )
    if distance is None:
        raise ValueError(
            "Distance must be provided with --distance or json_metadata['distance']."
        )
    return int(distance)


def build_relay_params(distance: int) -> dict:
    if distance not in RELAY_MEMORY_STRENGTHS:
        raise ValueError(f"Please specify a memory strength for distance {distance}.")

    return {
        "alpha": 1,
        "gamma0": 0.15000000000000002,
        "gamma_dist_interval": RELAY_MEMORY_STRENGTHS[distance],
        "pre_iter": 80,
        "num_sets": 60,
        "set_max_iter": 60,
        "stop_nconv": 1,
    }


def build_relay_bp_arguments(distance: int) -> dict:
    return {
        "custom_decoders": {
            "relay-bp": SinterDecoder_RelayBP(**build_relay_params(distance)),
        },
    }


def build_sliding_relay_bp_arguments(distance: int) -> dict:
    relay_params = build_relay_params(distance)
    decoder = decoders.SlidingWindowDecoder(
        distance * 2, distance // 2, with_RBP=True, **relay_params
    )
    print("Using sliding window relay BP decoder")
    return {"custom_decoders": {"sliding-window-relay-bp": decoder}}


def build_bposd_arguments() -> dict:
    return {
        "decoders": ["bposd"],
        "custom_decoders": {
            "bposd": LDPCSinterBpOsdDecoder(
                max_iter=10,
                osd_order=10,
                bp_method="ms",
                schedule="parallel",
            ),
        },
    }


def build_decoder_arguments(args: argparse.Namespace, json_metadata: dict) -> dict:
    if args.decoder is None:
        return {}

    if args.decoder == "pymatching":
        return {"decoders": ["pymatching"]}

    if args.decoder == "bposd":
        return build_bposd_arguments()

    distance = get_distance(args, json_metadata)
    if args.decoder == "relay-bp":
        return build_relay_bp_arguments(distance)
    if args.decoder == "sliding-relay-bp":
        return build_sliding_relay_bp_arguments(distance)

    raise ValueError(f"Unsupported decoder: {args.decoder}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a generated quantum circuit simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--circuit", type=str, required=True, help="Circuit file")
    parser.add_argument(
        "--output",
        type=str,
        default="results.pkl",
        help="The file to save the results to",
    )
    parser.add_argument(
        "--max_shots",
        type=int,
        default=1_000_000,
        help="Maximum number of shots for each simulation task",
    )
    parser.add_argument(
        "--max_errors",
        type=int,
        default=5_000,
        help="Maximum number of errors to collect before stopping",
    )
    parser.add_argument(
        "--json_metadata", type=str, default="{}", help="Json metadata for the task"
    )
    parser.add_argument(
        "--json_metadata_file",
        type=str,
        default=None,
        help="Json metadata file for the task",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers for simulation",
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=None,
        help="Code distance. Defaults to json_metadata['distance']",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        choices=["pymatching", "relay-bp", "bposd", "sliding-relay-bp"],
        help="The decoder to run",
    )
    parser.add_argument(
        "--save_resume_filepath",
        type=str,
        default=None,
        help=(
            "CSV file that shots are saved to incrementally while collecting. "
            "If it already exists (e.g. from a killed/retried run), collection "
            "resumes from the shots/errors already recorded there instead of "
            "starting over. Defaults to the same file the results are written "
            "to at the end (derived from --output)."
        ),
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable save/resume entirely, even for the default filepath.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.json_metadata_file is not None:
        with open(args.json_metadata_file) as f:
            json_metadata = json.load(f)
    else:
        json_metadata = json.loads(args.json_metadata)
    decoder_arguments = build_decoder_arguments(args, json_metadata)

    filename = os.path.splitext(os.path.basename(args.output))[0]
    csv_path = f"{filename}.csv"
    save_resume_filepath = None
    if not args.no_resume:
        save_resume_filepath = args.save_resume_filepath or csv_path

    result = run_simulations(
        circuit=args.circuit,
        json_metadata=json_metadata,
        num_workers=args.num_workers,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        save_resume_filepath=save_resume_filepath,
        **decoder_arguments,
    )

    with open(csv_path, "w") as file:
        file.write(sinter.CSV_HEADER + "\n")
        for task in result:
            file.write(task.to_csv_line() + "\n")


if __name__ == "__main__":
    main()
