#pragma once
#include "vec3.h"
// #include "path.h"
// #include <cuda_runtime.h>

/*
    These are specific cuda routines invoked by other functions
*/

// Test function
void test_func();

void execute_test_kernel(int nx, int ny, Vec3 *host_arr1, Vec3 *host_arr2, Vec3 *host_arr3);