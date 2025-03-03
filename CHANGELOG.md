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

Dev notes:
    - gpu option currently doesn't work (it is just a blank function call)
    - cuda_debug has been created and now does tensor addition on the cpu and the gpu
        - the gpu is much slower for small number of additions, but much faster for large additions
    - cuda_classes has been created and contains a special Vec3 class for use in the gpu