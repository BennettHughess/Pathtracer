# TO DO

- Optimize existing codebase
- Run the code on the GPU
- Make the codebase play nicer with different metrics
    - Let each metric have their own collision conditions
        - It might be concievable to have prebuild "scenarios" with their own collision conditions and metrics
    - Let each metric have their own `get_image_coords` function (this is necessary because you may need to convert from arbitrary coordinates to spherical coordinates)
- Implement image handling with PNG++ over PPMs (https://www.nongnu.org/pngpp/doc/0.2.1/)
- Add a script to create video
