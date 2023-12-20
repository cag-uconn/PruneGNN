from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='prune_gnn',
    ext_modules=[
        CUDAExtension('prune_gnn', [
            'prune_gnn.cc',
            'prune_spmm.cu',
            'cusparse_spmm_row.cu',
            'cublas_gemm.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })