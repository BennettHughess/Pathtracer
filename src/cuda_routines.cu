#include "../include/cuda_routines.h"
#include "../include/cuda_classes.cuh"
#include <iostream>

/*
    This .cu file needs to be compiled with the nvcc compiler.
*/

__global__ void test_kernel(size_t pitch1, size_t pitch2, size_t pitch3, int nx, int ny, CudaVec3 *arr1, CudaVec3 *arr2, CudaVec3 *arr3) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    CudaVec3* row1 = (CudaVec3*)((char*)arr1 + i * pitch1);
    CudaVec3* row2 = (CudaVec3*)((char*)arr2 + i * pitch2);
    CudaVec3* row3 = (CudaVec3*)((char*)arr3 + i * pitch3);

    if (i < ny && j < nx) {
        for (int k = 0; k < 1e8; k++) {
            row3[j] = row1[j] + row2[j]; 
        }
    }

}

void test_func() {
}

void execute_test_kernel(int nx, int ny, Vec3 *host_arr1, Vec3 *host_arr2, Vec3 *host_arr3) {

    CudaVec3* arr1 = new CudaVec3[nx*ny];
    CudaVec3* arr2 = new CudaVec3[nx*ny];
    CudaVec3* arr3 = new CudaVec3[nx*ny];

    // Convert host arrays into CudaVec arrays
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            arr1[i*nx + j] = Vec3_to_CudaVec3(host_arr1[i*nx + j]);
            arr2[i*nx + j] = Vec3_to_CudaVec3(host_arr2[i*nx + j]);
            arr3[i*nx + j] = Vec3_to_CudaVec3(host_arr3[i*nx + j]);
        }
    }

    // set up arrays and their pointers
    CudaVec3* dev_arr1; 
    CudaVec3* dev_arr2;
    CudaVec3* dev_arr3;
    size_t arr1_pitch;
    size_t arr2_pitch;
    size_t arr3_pitch;

    // allocate device memory for each array - sets up arr1_ptr and arr1_pitch
    cudaMallocPitch(&dev_arr1, &arr1_pitch, nx*sizeof(CudaVec3), ny);
    cudaMallocPitch(&dev_arr2, &arr2_pitch, nx*sizeof(CudaVec3), ny);
    cudaMallocPitch(&dev_arr3, &arr3_pitch, nx*sizeof(CudaVec3), ny);

    // copy host arrays onto device memory
    cudaMemcpy2D(dev_arr1, arr1_pitch, arr1, nx*sizeof(CudaVec3), nx*sizeof(CudaVec3), ny, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dev_arr2, arr2_pitch, arr2, nx*sizeof(CudaVec3), nx*sizeof(CudaVec3), ny, cudaMemcpyHostToDevice);
    cudaMemcpy2D(dev_arr3, arr3_pitch, arr3, nx*sizeof(CudaVec3), nx*sizeof(CudaVec3), ny, cudaMemcpyHostToDevice);

    // run kernel
    dim3 threadsPerBlock(10, 10);
    dim3 numBlocks(nx / threadsPerBlock.x, ny / threadsPerBlock.y);
    test_kernel<<<numBlocks, threadsPerBlock>>>(arr1_pitch, arr2_pitch, arr3_pitch, nx, ny, dev_arr1, dev_arr2, dev_arr3);
    cudaDeviceSynchronize();

    // copy device arrays back onto host
    cudaMemcpy2D(arr1, nx*sizeof(CudaVec3), dev_arr1, arr1_pitch, nx*sizeof(CudaVec3), ny, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(arr2, nx*sizeof(CudaVec3), dev_arr2, arr2_pitch, nx*sizeof(CudaVec3), ny, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(arr3, nx*sizeof(CudaVec3), dev_arr3, arr3_pitch, nx*sizeof(CudaVec3), ny, cudaMemcpyDeviceToHost);

    // save cuda arrays as normal vecs
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            host_arr1[i*nx + j] = CudaVec3_to_Vec3(arr1[i*nx + j]);
            host_arr2[i*nx + j] = CudaVec3_to_Vec3(arr2[i*nx + j]);
            host_arr3[i*nx + j] = CudaVec3_to_Vec3(arr3[i*nx + j]);
        }
    }

    // unallocate memory
    cudaFree(dev_arr1);
    cudaFree(dev_arr2);
    cudaFree(dev_arr3);

    delete[] arr1;
    delete[] arr2;
    delete[] arr3;

}

