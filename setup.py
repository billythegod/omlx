import os
import sys

from setuptools import setup


CUSTOM_KERNEL_FLAG = "--with-custom-kernel"
TRUTHY = {"1", "true", "yes", "on"}


def _with_custom_kernel() -> bool:
    if CUSTOM_KERNEL_FLAG in sys.argv:
        sys.argv.remove(CUSTOM_KERNEL_FLAG)
        return True
    return os.environ.get("OMLX_WITH_CUSTOM_KERNEL", "").strip().lower() in TRUTHY


def _custom_kernel_build_kwargs() -> dict:
    if not _with_custom_kernel():
        return {}

    from mlx import extension

    return {
        "ext_modules": [
            extension.CMakeExtension(
                "omlx.custom_kernels.glm_moe_dsa._ext",
                sourcedir="omlx/custom_kernels/glm_moe_dsa/csrc",
            )
        ],
        "cmdclass": {"build_ext": extension.CMakeBuild},
    }


if __name__ == "__main__":
    setup(**_custom_kernel_build_kwargs())
