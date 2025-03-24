Version 0.4.6

Changes:

- Added stricter compiler flags
- Cleaned up some code
- Added some error handling
- Added some better filesystem handling
    - Okay, it doesn't really work that well
- Changed .gitignore
- Updated make clean to also clean out binaries
- Changed the `Camera` class to allow for three different types of parallel types: 0 (cpu), 1 (cpu, multi), 2 (gpu)
- `cuda_debug` has been created and now does tensor addition on the cpu and the gpu
    - the gpu is much slower for small number of additions, but much faster for large additions
- `cuda_classes.cuh` has been created and contains special Vec3, Vec4, Path classes for use in the gpu
- Added GPU processing!
- Implemented a `Scenario` class that encapsulates the metric and redid the backend so different collision conditions can be passed in the future
- `cuda_misc.cuh` has been created, and currently only contains one function (but may include more later on).
- `cuda_metric.cuh` has been created and contains a CUDA-suitable metric, which takes calls from the config file.