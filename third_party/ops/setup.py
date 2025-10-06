from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_ops',
    ext_modules=[
        CUDAExtension(
            name='bias_act',
            sources=['bias_act.cpp'],
        ),
        CUDAExtension(
            name='gridsample_cuda',
            sources=[
                'gridsample_bindings.cpp',  # rename this file from grid_sample.cpp to avoid clash
                'gridsample_cuda.cu'
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
