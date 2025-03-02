#include "../include/cuda_routines.cuh"

/*
    This .cu file needs to be compiled with the nvcc compiler.
*/

void test_func() {
}

/*
__global__ void cuda_pathtrace(std::function<bool(Path&)> condition, const double dlam, Metric& metric, 
    int image_height, int image_width, std::vector<std::vector<Path>>& paths) {

    
    // Loop through image
    for (int i {0}; i < image_height; ++i) {

        // Progress bar
        std::clog << "\rPathtrace is " << int(100*(double(i)/image_height)) << "% completed. " 
            << "Working on row " << i << " of " << image_height << "." << std::flush;

        for (int j = 0; j < image_width; ++j) {

            // Trace a ray till it collides
            paths[i][j].loop_propagate(condition, dlam, metric);

        }
    }
    

}
*/