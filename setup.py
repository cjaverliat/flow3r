from setuptools import setup

try:
    from torch import cuda
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    all_cuda_archs = cuda.get_gencode_flags().replace('compute=', 'arch=').split()
    ext_modules = [
        CUDAExtension(
            name='curope',
            sources=[
                'src/flow3r/models/curope/curope.cpp',
                'src/flow3r/models/curope/kernels.cu',
            ],
            extra_compile_args=dict(
                nvcc=['-O3', '--ptxas-options=-v', '--use_fast_math'] + all_cuda_archs,
                cxx=['-O3']
            )
        )
    ]
    cmdclass = {'build_ext': BuildExtension}
except (OSError, ImportError):
    print("Unable to build curope. Skipping.")
    ext_modules = []
    cmdclass = {}

setup(ext_modules=ext_modules, cmdclass=cmdclass)
