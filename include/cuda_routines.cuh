#pragma once
// #include "path.h"
// #include <cuda_runtime.h>

/*
    These are specific cuda routines invoked by other functions
*/

// Test function
void test_func();

/*
// Utilized by the Camera class
__global__ void cuda_pathtrace(std::function<bool(Path&)> condition, const double dlam, Metric& metric, 
    int image_height, int image_width, std::vector<std::vector<Path>>& paths);
*/