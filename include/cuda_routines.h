#pragma once
#include "vec3.h"
#include "path.h"

/*
    These are specific cuda routines invoked by other functions
*/

// Pathtracing function with GPU parallelization
void cuda_pathtrace(std::function<bool(Path&)> condition, const double dlam, Metric& metric, 
    std::vector<std::vector<Path>> &paths, int image_height, int image_width);

// Test function
void execute_test_kernel(int nx, int ny, Vec3 *host_arr1, Vec3 *host_arr2, Vec3 *host_arr3);