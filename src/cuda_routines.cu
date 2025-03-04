#include "../include/cuda_routines.h"
#include "../include/cuda_classes.cuh"
#include <iostream>

/*
    This .cu file needs to be compiled with the nvcc compiler.
*/

/*********************************************************************************/
// Code for pathtracing with CUDA

// Kernels invoked by cuda_pathtrace
__global__ void pathtrace_kernel(CudaPath *paths, size_t pitch, int image_width, int image_height, double dlam) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // absolute incomprehensible code to get paths[i][j]
    CudaPath& path = ((CudaPath*)((char*)paths + i * pitch))[j];

    // propagate path!
    if (i < image_height && j < image_width) {
    
            path.loop_propagate(dlam, i, j);
        
    }

}

// Function invoked by non-cuda functions
void cuda_pathtrace(std::function<bool(Path&)> condition, const double dlam, Metric& metric, 
    std::vector<std::vector<Path>> &paths, int image_height, int image_width) {

    // declare array of cuda paths to be stored on the host and get pitch
    // host_pitch is used later when copying memory
    CudaPath* host_paths = new CudaPath[image_width*image_height];
    size_t host_pitch = image_width*sizeof(CudaPath);

    std::cout<< "CUDA: transfering paths into cudapath array..." << std::endl;

    // transfer paths object into CudaPath array
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            
            //std::cout << "path " << i << " " << j << ":     "<< paths[i][j].get_velocity() << std::endl;
            
            host_paths[i*image_width + j].set_position(
                Vec4_to_CudaVec4(paths[i][j].get_position())
            );
            host_paths[i*image_width + j].set_velocity(
                Vec4_to_CudaVec4(paths[i][j].get_velocity())
            );
        }
    }

    // declare array of cuda paths to be used on device
    CudaPath* dev_paths; 
    size_t dev_pitch;

    std::cout<< "CUDA: allocating and copying memory on device..." << std::endl;

    // allocate device memory for each array - this configures the dev_paths array and the dev_pitch size
    cudaMallocPitch(&dev_paths, &dev_pitch, image_width*sizeof(CudaPath), image_height);

    // copy host arrays onto device memory
    cudaMemcpy2D(dev_paths, dev_pitch, host_paths, host_pitch, image_width*sizeof(CudaPath), image_height, cudaMemcpyHostToDevice);

    std::cout<< "CUDA: executing kernel..." << std::endl;

    // run kernel
    dim3 threadsPerBlock(10, 10);
    dim3 numBlocks(image_width / threadsPerBlock.x, image_height / threadsPerBlock.y);
    pathtrace_kernel<<<numBlocks, threadsPerBlock>>>(dev_paths, dev_pitch, image_width, image_height, dlam);
    cudaDeviceSynchronize();

    std::cout<< "CUDA: transfering back to host..." << std::endl;

    // copy device arrays back onto host
    cudaMemcpy2D(host_paths, host_pitch, dev_paths, dev_pitch, image_width*sizeof(CudaPath), image_height, cudaMemcpyDeviceToHost);

    // transfer CudaPath array back onto paths object
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {

            paths[i][j].set_position(
                CudaVec4_to_Vec4(host_paths[i*image_width + j].get_position())
            );
            paths[i][j].set_velocity(
                CudaVec4_to_Vec4(host_paths[i*image_width + j].get_velocity())
            );

        }
    }
    
    // deallocate cuda arrays on device and host
    cudaFree(dev_paths);
    delete[] host_paths;

    std::cout<< "CUDA: finished." << std::endl;

}

/*********************************************************************************/
// Code for cuda_debug.cpp

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

