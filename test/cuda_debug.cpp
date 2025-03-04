#include <iostream>
#include "../include/vec3.h"
#include "../include/cuda_routines.h"

void cpu_add (int nx, int ny, Vec3* host_arr1, Vec3* host_arr2, Vec3* host_arr3) {
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            for (int k = 0; k < 1e8; k++) {
                host_arr3[i*nx + j] = host_arr1[i*nx + j] + host_arr2[i*nx + j];
            }
        }
    }
}

int main () {

    int nx = int(100);
    int ny = int(100);

    // initialize arrays
    Vec3* host_arr1 = new Vec3[nx*ny];
    Vec3* host_arr2 = new Vec3[nx*ny];
    Vec3* host_arr3 = new Vec3[nx*ny];

    std::cout << "initializing arrays..." << std::endl;

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            host_arr1[i*nx + j] = {1.*i, 2.*i, 3.*i};
            host_arr2[i*nx + j] = {1., 2., 3.};
            host_arr3[i*nx + j] = {0.,0.,0.};
        }
    }

    std::cout << "adding arrays..." << std::endl;

    // add (on cpu)
    //cpu_add(nx, ny, host_arr1, host_arr2, host_arr3);

    // add (on gpu)
    execute_test_kernel(nx, ny, host_arr1, host_arr2, host_arr3);

    /*
    std::cout << "printing arrays..." << std::endl;

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            std::cout << host_arr3[i*nx + j] << "       ";
        }
        std::cout << std::endl;
    }
    */
    
    std::cout << "deallocating arrays..." << std::endl;
    
    delete[] host_arr1;
    delete[] host_arr2;
    delete[] host_arr3;

    return 0;
}