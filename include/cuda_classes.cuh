#pragma once
#include "vec3.h"

// define class CudaVec3
class CudaVec3 {
    private:
    
        // define vectors as having an array of three doubles (the components)
        double e[3];
    
    public:
        // Constructors
        __host__ __device__ CudaVec3() : e{0,0,0} {}
        __host__ __device__ CudaVec3(double e0, double e1, double e2) : e{e0, e1, e2} {}
        __host__ __device__ CudaVec3(double v[3]) : e { v[0], v[1], v[2] } {}

        // Copy constructor
        __host__ __device__ CudaVec3(const CudaVec3& v) : e{v[0], v[1], v[2]} {}

        //Function forward declarations
        __host__ __device__ CudaVec3 operator-() const;

        __host__ __device__ double operator[](int i) const;

        __host__ __device__ CudaVec3& operator+=(CudaVec3& v);

        __host__ __device__ CudaVec3& operator*=(double t);

        __host__ __device__ CudaVec3& operator=(const CudaVec3& other);

};

// More forward declarations
__host__ __device__ CudaVec3 operator+(const CudaVec3& v1, const CudaVec3& v2);

// Subtract two vectors and store as new vector
__host__ __device__ CudaVec3 operator-(const CudaVec3& v1, const CudaVec3& v2);

// Multiply vector by scalar and save as new vector (made commutative)
__host__ __device__ CudaVec3 operator*(const CudaVec3& v, double t);
__host__ __device__ CudaVec3 operator*(double t, const CudaVec3& v);

// Convert to and from CudaVec
__host__ Vec3 CudaVec3_to_Vec3(CudaVec3 v);
__host__ CudaVec3 Vec3_to_CudaVec3(Vec3 v);