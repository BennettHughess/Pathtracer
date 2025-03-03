#include "../include/cuda_classes.cuh"

/*
    CUDA VECTOR OPERATIONS 	ʕっ•ᴥ•ʔっ
*/

// Negate a vector
__host__ __device__ CudaVec3 CudaVec3::operator-() const { 
    return CudaVec3(-e[0], -e[1], -e[2]); 
}

// Access vector as an array
__host__ __device__ double CudaVec3::operator[](int i) const {
    // note: no exception handling for out-of-bounds indices
    return e[i];
}

// Add vector to existing vector
__host__ __device__ CudaVec3& CudaVec3::operator+=(CudaVec3& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

// Multiply vector by scalar
__host__ __device__ CudaVec3& CudaVec3::operator*=(double t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

// Set two vectors equal
__host__ __device__ CudaVec3& CudaVec3::operator=(const CudaVec3& v) {
    e[0] = v.e[0];
    e[1] = v.e[1];
    e[2] = v.e[2];
    return *this;
}

/*
    CONVENIENT BINARY VECTOR OPERATIONS
*/

// Add two vectors and store as new vector
__host__ __device__ CudaVec3 operator+(const CudaVec3& v1, const CudaVec3& v2) {
    return CudaVec3(v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2]);
}

// Subtract two vectors and store as new vector
__host__ __device__ CudaVec3 operator-(const CudaVec3& v1, const CudaVec3& v2) {
    return v1 + (-v2);
}

// Multiply vector by scalar and save as new vector (made commutative)
__host__ __device__ CudaVec3 operator*(const CudaVec3& v, double t) {
    return CudaVec3(v[0]*t, v[1]*t, v[2]*t);
}
__host__ __device__ CudaVec3 operator*(double t, const CudaVec3& v) {
    return v*t;
}

// Convert to and from CudaVec
__host__ Vec3 CudaVec3_to_Vec3(CudaVec3 v) {
    return Vec3{v[0], v[1], v[2]};
}
__host__ CudaVec3 Vec3_to_CudaVec3(Vec3 v){
    return CudaVec3{v[0], v[1], v[2]};
}