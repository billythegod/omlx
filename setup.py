from setuptools import setup

from mlx import extension


if __name__ == "__main__":
    setup(
        ext_modules=[
            extension.CMakeExtension(
                "omlx_glm_kernels._ext",
                sourcedir="omlx_glm_kernels/csrc",
            )
        ],
        cmdclass={"build_ext": extension.CMakeBuild},
    )
