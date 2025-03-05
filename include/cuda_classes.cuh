#pragma once
#include "vec3.h"
#include "vec4.h"

/****************************************************/
// Vec3 for CUDA
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

        __host__ __device__ double norm() const;

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

/****************************************************/
// Vec4 for CUDA (identical to 3vec)
class CudaVec4 {
    private:
    
        // define vectors as having an array of three doubles (the components)
        double e[4];
    
    public:
        // Constructors
        __host__ __device__ CudaVec4() : e{0,0,0,0} {}
        __host__ __device__ CudaVec4(double e0, double e1, double e2, double e3) : e{e0, e1, e2, e3} {}
        __host__ __device__ CudaVec4(double v[4]) : e { v[0], v[1], v[2], v[3] } {}

        // Copy constructor
        __host__ __device__ CudaVec4(const CudaVec4& v) : e{v[0], v[1], v[2], v[3]} {}

        //Function forward declarations
        __host__ __device__ CudaVec4 operator-() const;

        __host__ __device__ double operator[](int i) const;

        __host__ __device__ CudaVec4& operator+=(CudaVec4& v);

        __host__ __device__ CudaVec4& operator*=(double t);

        __host__ __device__ CudaVec4& operator=(const CudaVec4& other);

        // Access functions
        __host__ __device__ CudaVec3 get_vec3() { return CudaVec3{e[1],e[2],e[3]}; }

};

// More forward declarations
__host__ __device__ CudaVec4 operator+(const CudaVec4& v1, const CudaVec4& v2);

// Subtract two vectors and store as new vector
__host__ __device__ CudaVec4 operator-(const CudaVec4& v1, const CudaVec4& v2);

// Multiply vector by scalar and save as new vector (made commutative)
__host__ __device__ CudaVec4 operator*(const CudaVec4& v, double t);
__host__ __device__ CudaVec4 operator*(double t, const CudaVec4& v);

// Convert to and from CudaVec
__host__ Vec4 CudaVec4_to_Vec4(CudaVec4 v);
__host__ CudaVec4 Vec4_to_CudaVec4(Vec4 v);

/****************************************************/
// Cuda paths stripped down

class CudaPath {

    public:
        
        enum Integrator {
            RK4,
            CashKarp
        };

    private:
        // Paths have a position and direction (4-vectors)
        CudaVec4 position;
        CudaVec4 velocity;

        // Paths need an integrator to integrate the geodesic equations
        Integrator integrator {CashKarp};

        // for adaptive integrators, we need a minimum step size, maximum step size, and tolerance
        double tolerance {1e-6};
        double min_dlam {1e-20};
        double max_dlam {5};

        // Propagation methods
        __host__ __device__ void rk4_propagate(double dlam);

        // Cash Karp
        __host__ __device__ double cashkarp_propagate(double dlam, CudaVec4* ylist);
        __host__ __device__ void cashkarp_integrate(double dlam, CudaVec4* ylist);

        // temporary function to get acceleration until metric class is implemented
        __host__ __device__ CudaVec4 get_acceleration(const CudaVec4& pos, const CudaVec4& vel);
        
    public:

        // Constructors
        __host__ __device__ CudaPath() : position {0,0,0,0}, velocity {1,0,0,0} {}
        __host__ __device__ CudaPath(const CudaVec4& pos, const CudaVec4& vel) : position {pos}, velocity {vel} {}

        // Access functions
        __host__ __device__ CudaVec4& get_position() {return position;}
        __host__ __device__ CudaVec4& get_velocity() {return velocity;}

        __host__ __device__ void set_integrator(Integrator integ) {integrator = integ;}
        __host__ __device__ void set_min_dlam(double min) {min_dlam = min;}
        __host__ __device__ void set_max_dlam(double max) {max_dlam = max;}
        __host__ __device__ void set_tolerance(double tol) {tolerance = tol;}

        __host__ __device__ void set_position(const CudaVec4& pos) {position = pos;}
        __host__ __device__ void set_velocity(const CudaVec4& vel) {velocity = vel;}

        // Propagate path until condition (which is a lambda function) is met
        // note: condition is currently not passed
        __host__ __device__ void loop_propagate(double dlam);
        __host__ __device__ double propagate(double dlam, CudaVec4* ylist);

};