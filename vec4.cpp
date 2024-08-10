#include <cmath>
#include <iostream>
#include <vector>
#include "vec4.h"

/*
    VECTOR OPERATIONS 	ʕっ•ᴥ•ʔっ
*/

// Negate a vector
Vec4 Vec4::operator-() const { 
    return Vec4 {-e[0], -e[1], -e[2], -e[3]}; 
}

// Access vector as an array
double Vec4::operator[](int i) const {
    // note: no exception handling for out-of-bounds indices
    return e[i];
}

// Add vector to existing vector
Vec4& Vec4::operator+=(Vec4& v) {
    e[0] += v[0];
    e[1] += v[1];
    e[2] += v[2];
    e[3] += v[3];
    return *this;
}

// Multiply vector by scalar
Vec4& Vec4::operator*=(double t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    e[3] *= t;
    return *this;
}

// Compute norm
double Vec4::norm_squared(Metric& metric, Vec4& position) {
    std::vector<double> components = metric.get_components(position);
    double norm_squared {
        components[0]*e[0]*e[0] + components[1]*e[1]*e[1] + components[2]*e[2]*e[2] + components[3]*e[3]*e[3]
    };
    return norm_squared; 
}

// compute max element of a vector
double Vec4::max() {

    double max {e[0]};

    for (int i {1}; i < 4; ++i) {
        max = std::max(e[i], max);
    }
    
    return max;

}

/*
    CONVENIENT BINARY VECTOR OPERATIONS
*/

// Add two vectors and store as new vector
Vec4 operator+(const Vec4& v1, const Vec4& v2) {
    return Vec4(v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2], v1[3]+v2[3]);
}

// Subtract two vectors and store as new vector
Vec4 operator-(const Vec4& v1, const Vec4& v2) {
    return v1 + (-v2);
}

// Multiply vector by scalar and save as new vector (made commutative)
Vec4 operator*(const Vec4& v, double t) {
    return Vec4(v[0]*t, v[1]*t, v[2]*t, v[3]*t);
}
Vec4 operator*(double t, const Vec4& v) {
    return v*t;
}

// Multiply vectors elementwise
Vec4 operator*(const Vec4& v1, const Vec4& v2) {
    return Vec4(v1[0]*v2[0], v1[1]*v2[1], v1[2]*v2[2], v1[3]*v2[3]);
}

// Output code
std::ostream& operator<<(std::ostream& out, const Vec4& v) {
    return out << v[0] << ' ' << v[1] << ' ' << v[2] << ' ' << v[3];
}

// Turn 3-vector into null vector in the same direction. May need position dependent metric
Vec4 convert_to_null(const Vec3& v, const Vec4& position, Metric& metric) {
    std::vector<double> met = metric.get_components(position);

    // for null vector with diagonal metric: v0^2 = (-1/g_00) v^i v_i
    double v0 { sqrt((-1/met[0])*(met[1]*v[0]*v[0] + met[2]*v[1]*v[1] + met[3]*v[2]*v[2])) };

    // return v as a 4-vector with v[0] scaled correctly
    return Vec4 { v0, v };
}

// take inner product of v1 and v2 given a position-dependent metric
double inner(const Vec4& v1, const Vec4& v2, Vec4& position, Metric& metric) {
    std::vector<double> met = metric.get_components(position);
    return v1[0]*v2[0]*met[0] + v1[1]*v2[1]*met[1] + v1[2]*v2[2]*met[2] + v1[3]*v2[3]*met[3];
}