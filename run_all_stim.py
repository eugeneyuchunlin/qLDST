#!/usr/bin/env python3
"""Run runner.py once for every .stim file in a directory.

Unlike run_jobs_locally.py (which replays HTCondor .sub files), this just
globs for *.stim circuits and calls runner.py directly, passing through
whatever decoder/shot/metadata options you give on the command line to
every circuit found.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def find_stim_files(directory: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.stim" if recursive else "*.stim"
    stim_files = sorted(directory.glob(pattern))
    if not stim_files:
        raise SystemExit(f"No .stim files found in {directory}")
    return stim_files


def build_command(
    runner_py: Path, circuit: Path, output_dir: Path, args: argparse.Namespace
) -> tuple[list[str], Path]:
    output_path = output_dir / f"{circuit.stem}.pkl"

    command = [
        sys.executable,
        str(runner_py),
        "--circuit",
        str(circuit),
        "--output",
        str(output_path),
        "--max_shots",
        str(args.max_shots),
        "--max_errors",
        str(args.max_errors),
        "--num_workers",
        str(args.num_workers),
    ]
    if args.decoder is not None:
        command += ["--decoder", args.decoder]
    if args.distance is not None:
        command += ["--distance", str(args.distance)]
    if args.json_metadata_file is not None:
        command += ["--json_metadata_file", args.json_metadata_file]
    else:
        metadata = json.loads(args.json_metadata)
        metadata["filename"] = circuit.name
        command += ["--json_metadata", json.dumps(metadata)]

    return command, output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to search for .stim files (default: current directory)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also search subdirectories for .stim files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write .pkl/.csv results to (default: same as directory)",
    )
    parser.add_argument(
        "--max_shots", type=int, default=1_000_000, help="Forwarded to runner.py"
    )
    parser.add_argument(
        "--max_errors", type=int, default=5_000, help="Forwarded to runner.py"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Forwarded to runner.py"
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default=None,
        choices=["pymatching", "relay-bp", "bposd", "sliding-relay-bp"],
        help="Forwarded to runner.py",
    )
    parser.add_argument(
        "--distance", type=int, default=None, help="Forwarded to runner.py"
    )
    parser.add_argument(
        "--json_metadata", type=str, default="{}", help="Forwarded to runner.py"
    )
    parser.add_argument(
        "--json_metadata_file",
        type=str,
        default=None,
        help="Forwarded to runner.py",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a circuit if its --output .pkl file already exists",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with the next circuit even if one fails (default: stop on failure)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them",
    )
    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else directory
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_py = Path(__file__).resolve().parent / "runner.py"

    stim_files = find_stim_files(directory, args.recursive)
    print(f"Found {len(stim_files)} .stim file(s) in {directory}")

    for circuit in stim_files:
        command, output_path = build_command(runner_py, circuit, output_dir, args)

        if args.skip_existing and output_path.exists():
            print(f"[skip] {output_path.name} already exists")
            continue

        print(f"[run] {circuit.name}: {' '.join(command)}")
        if args.dry_run:
            continue

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[fail] {circuit.name}: exited with {e.returncode}")
            if not args.keep_going:
                raise


if __name__ == "__main__":
    main()
