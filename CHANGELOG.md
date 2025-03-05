Version 0.4.6

Changes:
    - Added stricter compiler flags
    - Cleaned up some code
    - Added some error handling
    - Added some better filesystem handling
        - Okay, it doesn't really work that well
    - Changed .gitignore
    - Updated make clean to also clean out binaries
    - Changed the Camera class to allow for three different types of parallel types: 0 (cpu), 1 (cpu, multi), 3 (gpu)
    - cuda_debug has been created and now does tensor addition on the cpu and the gpu
        - the gpu is much slower for small number of additions, but much faster for large additions
    - cuda_classes has been created and contains special Vec3, Vec4, Path classes for use in the gpu
    - Added GPU processing!

dev notes:
    - gpu processing is still very barebones, and needs to be better integrated into the rest of the codebase
        - I have implemented passing tolerance, min_dlam, max_dlam to the cuda paths, but still need to pass collision conditions and integrator type
        - i also need to provide explicit type checking to make sure nothing is passed that shouldnt be
    - need to provide more dynamic thread allocation
    - i also need to look into what kind of speedup i can get from passing everything as a float