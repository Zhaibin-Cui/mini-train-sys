"""Execute a server notebook into an audited output copy.

The source notebook is never modified. Cell errors are captured in the output
copy, validated after execution, and cause a non-zero exit. A successful result
is published with an atomic rename so interrupted runs cannot look complete.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _cell_errors(path: Path) -> list[dict[str, object]]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(
                    {
                        "cell": cell_index,
                        "ename": output.get("ename"),
                        "evalue": output.get("evalue"),
                        "traceback": output.get("traceback", [])[-8:],
                    }
                )
    return errors


def execute_notebook(
    notebook: str | Path,
    *,
    output_dir: str | Path,
    kernel: str,
    timeout_seconds: int,
) -> Path:
    source = Path(notebook)
    if not source.is_absolute():
        source = ROOT / source
    if not source.is_file():
        raise FileNotFoundError(source)
    destination_dir = Path(output_dir)
    if not destination_dir.is_absolute():
        destination_dir = ROOT / destination_dir
    destination_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    staging = destination_dir / f".{source.stem}_executed_{timestamp}.tmp.ipynb"
    final = destination_dir / f"{source.stem}_executed_{timestamp}.ipynb"
    log = destination_dir / f"{source.stem}_executed_{timestamp}.log"
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--allow-errors",
        f"--ExecutePreprocessor.kernel_name={kernel}",
        f"--ExecutePreprocessor.timeout={timeout_seconds}",
        "--output",
        staging.name,
        "--output-dir",
        str(destination_dir),
        str(source),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    log.write_text(
        "COMMAND\n"
        + subprocess.list2cmdline(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0 or not staging.is_file():
        raise RuntimeError(
            f"notebook process failed with exit {completed.returncode}; inspect {log}"
        )
    errors = _cell_errors(staging)
    if errors:
        failed = destination_dir / f"{source.stem}_failed_{timestamp}.ipynb"
        staging.replace(failed)
        error_path = failed.with_suffix(".errors.json")
        error_path.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(
            f"notebook captured {len(errors)} cell error(s); inspect {failed} and {error_path}"
        )
    staging.replace(final)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    parser.add_argument("--output-dir", default="artifacts/notebooks")
    parser.add_argument("--kernel", default="mini-train-sys")
    parser.add_argument("--timeout-seconds", type=int, default=-1)
    args = parser.parse_args()
    print(
        execute_notebook(
            args.notebook,
            output_dir=args.output_dir,
            kernel=args.kernel,
            timeout_seconds=args.timeout_seconds,
        )
    )


if __name__ == "__main__":
    main()
