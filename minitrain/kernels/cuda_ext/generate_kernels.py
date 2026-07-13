"""Generate the explicit CUDA template instantiations used by cuda_ext.

FlashAttention keeps the expensive kernel implementation in common headers and
places each dtype/head-dimension/causal combination in a tiny ``.cu`` file.
That layout lets ninja compile configurations independently and prevents one
nvcc process from retaining the complete template matrix in memory.

MiniTrain intentionally generates a smaller matrix than upstream:

* dense attention only;
* forward and backward;
* fp16 and bf16;
* head-dimension buckets 32, 64, 96, 128, 192, and 256;
* causal and non-causal masks.

Dropout is not a filename dimension. The upstream launch template uses
``DROPOUT_SWITCH`` inside each translation unit, so nvcc emits separate
``Is_dropout=true`` and ``Is_dropout=false`` kernels. In particular, the false
specialization compiles out Philox state, random-number generation, and the
dropout mask rather than paying for a runtime branch in the hot loop.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path


# Keep this matrix close to upstream FlashAttention 2.8.4. Head dimensions are
# buckets: for example D=80 dispatches to the 96 specialization with masked
# loads/stores for the uneven tail.
DTYPES = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}
HEAD_DIMENSIONS = (32, 64, 96, 128, 192, 256)
DIRECTIONS = ("fwd", "bwd")
CAUSAL_MODES = (False, True)


@dataclass(frozen=True)
class KernelInstantiation:
    """One independently compiled point in the kernel dispatch matrix."""

    direction: str
    dtype: str
    head_dim: int
    is_causal: bool

    @property
    def filename(self) -> str:
        causal = "_causal" if self.is_causal else ""
        return f"flash_{self.direction}_hdim{self.head_dim}_{self.dtype}{causal}_sm80.cu"

    def render(self) -> str:
        """Render the same thin specialization pattern used by upstream."""

        scalar = DTYPES[self.dtype]
        causal = "true" if self.is_causal else "false"
        launch_header = f"flash_{self.direction}_launch_template.h"
        public_name = f"run_mha_{self.direction}_"
        dispatch_name = f"run_mha_{self.direction}_hdim{self.head_dim}"
        return f"""// Copyright (c) 2024, Tri Dao.
// Adapted for mini-train-sys from FlashAttention 2.8.4 generate_kernels.py.
// This file is generated. Edit cuda_ext/generate_kernels.py instead.

#include "namespace_config.h"
#include "{launch_header}"

namespace FLASH_NAMESPACE {{

// Bind one public dispatch symbol to one fully compile-time-specialized tree.
template <>
void {public_name}<{scalar}, {self.head_dim}, {causal}>(
    Flash_{self.direction}_params& params,
    cudaStream_t stream) {{
    {dispatch_name}<{scalar}, {causal}>(params, stream);
}}

}}  // namespace FLASH_NAMESPACE
"""


def all_instantiations() -> list[KernelInstantiation]:
    """Return the deterministic source matrix consumed by the build loader."""

    return [
        KernelInstantiation(direction, dtype, head_dim, is_causal)
        for direction, dtype, head_dim, is_causal in itertools.product(
            DIRECTIONS, DTYPES, HEAD_DIMENSIONS, CAUSAL_MODES
        )
    ]


def generate(output_dir: Path, *, check: bool) -> None:
    """Write generated sources, or verify checked-in files with ``--check``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    stale_or_missing: list[str] = []

    # Generate every matrix point independently so diffs remain reviewable and
    # ninja can parallelize compilation on larger build servers.
    for kernel in all_instantiations():
        expected_names.add(kernel.filename)
        path = output_dir / kernel.filename
        content = kernel.render()
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale_or_missing.append(kernel.filename)
        else:
            # The rendered string already contains explicit LF newlines. Avoid
            # Path.write_text(newline=...), which is unavailable on the older
            # Python version used by the local CUDA workstation.
            path.write_text(content, encoding="utf-8")

    # Refuse orphaned generated files. Otherwise removing a matrix point from
    # the generator would silently leave the old CUDA object in future builds.
    unexpected = sorted(path.name for path in output_dir.glob("*.cu") if path.name not in expected_names)
    if check and (stale_or_missing or unexpected):
        details = [*(f"stale/missing: {name}" for name in stale_or_missing), *(f"unexpected: {name}" for name in unexpected)]
        raise SystemExit("Generated CUDA sources are out of date:\n" + "\n".join(details))
    if not check:
        for name in unexpected:
            (output_dir / name).unlink()


def main() -> None:
    """CLI entry point used by developers and CI."""

    default_output = Path(__file__).resolve().parent / "csrc" / "instantiations"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generate(args.output_dir, check=args.check)


if __name__ == "__main__":
    main()
