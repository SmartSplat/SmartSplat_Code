from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

setup(
    name="simple_knn_2d_qr",
    ext_modules=[
        CUDAExtension(
            "simple_knn_2d_qr._C",
            [
                "simple_knn_2d_qr/ext.cpp",
                "simple_knn_2d_qr/simple_knn_2d_qr.cu",
            ],
            extra_compile_args={
                "nvcc": [
                    "--extended-lambda",
                    "-O3",
                ],
                "cxx": cxx_compiler_flags + ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
