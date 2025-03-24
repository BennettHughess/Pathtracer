#include "../include/cuda_classes.cuh"

/*****************************************************************************/
/**********************     Cuda path class      *****************************/

__device__ void CudaPath::rk4_propagate(double dlam, CudaMetric& metric) {

    // we have two systems of equations: x'' = accel, and x' = vel.
    // first we get the current position and velocities. let y1 = pos, y2 = vel.
    CudaVec4 y1 { get_position() };
    CudaVec4 y2 { get_velocity() };

    // first step: getting k1 (for the y2 vector) and l1 (for the y1 vector)
    CudaVec4 l1 { dlam*y2 };
    CudaVec4 k1 { dlam*metric.get_acceleration(y1, y2) };

    // second step: getting k2 and l2
    CudaVec4 l2 { dlam*(y2 + 0.5*l1) };
    CudaVec4 k2 { dlam*metric.get_acceleration(y1 + 0.5*l1, y2 + 0.5*k1) };

    // third step: getting k3 and l3
    CudaVec4 l3 { dlam*(y2 + 0.5*l2) };
    CudaVec4 k3 { dlam*metric.get_acceleration(y1 + 0.5*l2, y2 + 0.5*k2) };

    // fourth step: getting k4 and l4
    CudaVec4 l4 { dlam*(y2 + l3) };
    CudaVec4 k4 { dlam*metric.get_acceleration(y1 + l3, y2 + k3) };

    // fifth step: update position and velocity with new values
    set_position(y1 + (double(1)/double(6))*(l1 + 2*l2 + 2*l3 + l4));
    set_velocity(y2 + (double(1)/double(6))*(k1 + 2*k2 + 2*k3 + k4));

}

// cash karp!!
__device__ double CudaPath::cashkarp_propagate(double dlam, CudaVec4* ylist, CudaMetric& metric) {

    // returns a list: {y1_order4, y1_order5, y2_order4, y2_order5}
    cashkarp_integrate(dlam, ylist, metric);

    // compute error: difference in the order 5 and order 4 solutions
    CudaVec4 y1_error { ylist[1] - ylist[0] };
    CudaVec4 y2_error { ylist[3] - ylist[2] };
    double err_list[] {y1_error[0], y1_error[1], y1_error[2], y1_error[3],
        y2_error[0], y2_error[1], y2_error[2], y2_error[3]};

    // find largest error and record which one it is
    double max_error {abs(err_list[0])};
    int index_error {0};
    for (int i {1}; i < 8; ++i) {
        if (abs(err_list[i]) >= max_error) {
            max_error = abs(err_list[i]);
            index_error = i;
        }
    }

    // get the new y value with the most error in it
    double y_max_error {0};
    if (index_error < 4) {
        y_max_error = abs(ylist[1][index_error]);
    }
    else {
        y_max_error = abs(ylist[3][index_error-4]);
    }

    // compute new step size
    // note: i think its better if the acceptable error is *fractional*, i.e. error = tolerance*y
    double best_dlam {
        0.9*dlam*pow( tolerance*y_max_error/max_error , 1.0/5)
    };

    double new_dlam {};
    if (best_dlam > max_dlam) {
        new_dlam = max_dlam;
    }
    else if (best_dlam < min_dlam) {
        new_dlam = min_dlam;
    }
    else {
        new_dlam = best_dlam;
    }


    if (dlam > new_dlam) {
        // reintegrate and use the new values
        cashkarp_integrate(new_dlam, ylist, metric);
    }

    // update position, velocity with the fifth order estimate
    set_position(ylist[1]);
    set_velocity(ylist[3]);

    return new_dlam;

}

// do one step of cashkarp integration
__device__ void CudaPath::cashkarp_integrate(double dlam, CudaVec4* ylist, CudaMetric& metric) {

    // proceed very similarly to rk45
    // let y1 = pos, y2 = vel
    CudaVec4 y1 { get_position() };
    CudaVec4 y2 { get_velocity() };

    // first step: k1, l1
    CudaVec4 l1 { dlam*y2 };
    CudaVec4 k1 { dlam*metric.get_acceleration(y1,y2) };

    // k2, l2
    CudaVec4 l2 { dlam*(y2 + (1.0/5)*l1) };
    CudaVec4 k2 { dlam*metric.get_acceleration(y1 + (1.0/5)*l1, y2 + (1.0/5)*k1) };

    // k3, l3
    CudaVec4 l3 { dlam*(y2 + (3.0/40)*l1 + (9.0/40)*l2) };
    CudaVec4 k3 { dlam*metric.get_acceleration(y1 + (3.0/40)*l1 + (9.0/40)*l2, y2 + (3.0/40)*k1 + (9.0/40)*k2) };

    // k4, l4
    CudaVec4 l4 { dlam*(y2 + (3.0/10)*l1 - (9.0/10)*l2 + (6.0/5)*l3) };
    CudaVec4 k4 { dlam*metric.get_acceleration(y1 + (3.0/10)*l1 - (9.0/10)*l2 + (6.0/5)*l3,
        y2 + (3.0/10)*k1 - (9.0/10)*k2 + (6.0/5)*k3) };

    // k5, l5
    CudaVec4 l5 { dlam*(y2 - (11.0/54)*l1 + (5.0/2)*l2 - (70.0/27)*l3 + (35.0/27)*l4) };
    CudaVec4 k5 { dlam*metric.get_acceleration(y1 - (11.0/54)*l1 + (5.0/2)*l2 - (70.0/27)*l3 + (35.0/27)*l4,
        y2 - (11.0/54)*k1 + (5.0/2)*k2 - (70.0/27)*k3 + (35.0/27)*k4) };

    // k6, l6
    CudaVec4 l6 { dlam*(y2 + (1631.0/55296)*l1 + (175.0/512)*l2 + (575.0/13824)*l3 + (44275.0/110592)*l4 + (253.0/4096)*l5) };
    CudaVec4 k6 { dlam*metric.get_acceleration(y1 + (1631.0/55296)*l1 + (175.0/512)*l2 + (575.0/13824)*l3 + (44275.0/110592)*l4 + (253.0/4096)*l5,
        y2 + (1631.0/55296)*k1 + (175.0/512)*k2 + (575.0/13824)*k3 + (44275.0/110592)*k4 + (253.0/4096)*k5) };

    // compute y1, y2
    CudaVec4 y1_order4 { y1 + (2825.0/27648)*l1 + (18575.0/48384)*l3 + (13525.0/55296)*l4 + (277.0/14336)*l5 + (1.0/4)*l6 };
    CudaVec4 y2_order4 { y2 + (2825.0/27648)*k1 + (18575.0/48384)*k3 + (13525.0/55296)*k4 + (277.0/14336)*k5 + (1.0/4)*k6 };
    CudaVec4 y1_order5 { y1 + (37.0/378)*l1 + (250.0/621)*l3 + (125.0/594)*l4 + (512.0/1771)*l6 };
    CudaVec4 y2_order5 { y2 + (37.0/378)*k1 + (250.0/621)*k3 + (125.0/594)*k4 + (512.0/1771)*k6 };

    ylist[0] = y1_order4;
    ylist[1] = y1_order5;
    ylist[2] = y2_order4;
    ylist[3] = y2_order5;

    //printf("KERNEL: ylist is {{%f,%f,%f,%f},    {%f,%f,%f,%f},  {%f,%f,%f,%f},  {%f,%f,%f,%f}} \n", ylist[0][0], ylist[0][1], ylist[0][2], ylist[0][3]
    //    , ylist[1][0], ylist[1][1], ylist[1][2], ylist[1][3], ylist[2][0], ylist[2][1], ylist[2][2], ylist[2][3], ylist[3][0], ylist[3][1], ylist[3][2], ylist[3][3]);

}

__device__ double CudaPath::propagate(double dlam, CudaVec4* ylist, CudaMetric& metric) {

    switch (integrator) {

        case Path::RK4:
            rk4_propagate(dlam, metric);
            break;

        case Path::CashKarp:
            dlam = cashkarp_propagate(dlam, ylist, metric);
            break;

        default:
            printf("CUDA: unacceptable integrator picked. Check list of allowed integrators.");
            break;

    }    
    return dlam;

}

// Propagate path until condition is no longer met
__device__ void CudaPath::loop_propagate(double dlam, Scenario::ScenarioType scenario_type, ScenarioParameters params, CudaMetric& metric) {

    // ylist to be overwritten (allocate here)
    CudaVec4 ylist[4] = {CudaVec4({0,0,0,0}),CudaVec4({0,0,0,0}),CudaVec4({0,0,0,0}),CudaVec4({0,0,0,0})};
        
    while (cuda_collision_checker(*this, scenario_type, params)) {
        dlam = propagate(dlam, ylist, metric);
    }

}


/*
    CUDA VECTOR OPERATIONS 	ʕっ•ᴥ•ʔっ
*/

/************************   CudaVec3 stuff    **********************/
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

// Compute norm
double CudaVec3::norm() const {
    return std::sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
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

// Convert to and from CudaVec HOST ONLY
__host__ Vec3 CudaVec3_to_Vec3(CudaVec3 v) {
    return Vec3{v[0], v[1], v[2]};
}
__host__ CudaVec3 Vec3_to_CudaVec3(Vec3 v){
    return CudaVec3{v[0], v[1], v[2]};
}


/************************   CudaVec4 stuff    **********************/
// Negate a vector
__host__ __device__ CudaVec4 CudaVec4::operator-() const { 
    return CudaVec4(-e[0], -e[1], -e[2], -e[3]); 
}

// Access vector as an array
__host__ __device__ double CudaVec4::operator[](int i) const {
    // note: no exception handling for out-of-bounds indices
    return e[i];
}

// Add vector to existing vector
__host__ __device__ CudaVec4& CudaVec4::operator+=(CudaVec4& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    e[3] += v.e[3];
    return *this;
}

// Multiply vector by scalar
__host__ __device__ CudaVec4& CudaVec4::operator*=(double t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    e[3] *= t;
    return *this;
}

// Set two vectors equal
__host__ __device__ CudaVec4& CudaVec4::operator=(const CudaVec4& v) {
    e[0] = v.e[0];
    e[1] = v.e[1];
    e[2] = v.e[2];
    e[3] = v.e[3];
    return *this;
}

/*
    CONVENIENT BINARY VECTOR OPERATIONS
*/

// Add two vectors and store as new vector
__host__ __device__ CudaVec4 operator+(const CudaVec4& v1, const CudaVec4& v2) {
    return CudaVec4(v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2], v1[3]+v2[3]);
}

// Subtract two vectors and store as new vector
__host__ __device__ CudaVec4 operator-(const CudaVec4& v1, const CudaVec4& v2) {
    return v1 + (-v2);
}

// Multiply vector by scalar and save as new vector (made commutative)
__host__ __device__ CudaVec4 operator*(const CudaVec4& v, double t) {
    return CudaVec4(v[0]*t, v[1]*t, v[2]*t, v[3]*t);
}
__host__ __device__ CudaVec4 operator*(double t, const CudaVec4& v) {
    return v*t;
}

// Convert to and from CudaVec HOST ONLY
__host__ Vec4 CudaVec4_to_Vec4(CudaVec4 v) {
    return Vec4{v[0], v[1], v[2], v[3]};
}
__host__ CudaVec4 Vec4_to_CudaVec4(Vec4 v){
    return CudaVec4{v[0], v[1], v[2], v[3]};
}