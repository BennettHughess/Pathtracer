#include <cmath>
#include <iostream>
#include "vec3.h"

/*
    VECTOR OPERATIONS 	ʕっ•ᴥ•ʔっ
*/

// Negate a vector
Vec3 Vec3::operator-() const { 
    return Vec3(-e[0], -e[1], -e[2]); 
}

// Access vector as an array
double Vec3::operator[](int i) const {
    // note: no exception handling for out-of-bounds indices
    return e[i];
}

// Add vector to existing vector
Vec3& Vec3::operator+=(Vec3& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

// Multiply vector by scalar
Vec3& Vec3::operator*=(double t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

// Compute norm
double Vec3::norm() const {
    return std::sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
}

/*
    CONVENIENT BINARY VECTOR OPERATIONS
*/

// Add two vectors and store as new vector
Vec3 operator+(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.e[0]+v2.e[0], v1.e[1]+v2.e[1], v1.e[2]+v2.e[2]);
}

// Subtract two vectors and store as new vector
Vec3 operator-(const Vec3& v1, const Vec3& v2) {
    return v1 + (-v2);
}

// Take dot product of two vectors and store as double
double dot(const Vec3& v1, const Vec3& v2) {
    return std::sqrt(v1.e[0]*v2.e[0] + v1.e[1]*v2.e[1] + v1.e[2]*v2.e[2]); 
}

// Multiply vector by scalar and save as new vector (made commutative)
Vec3 operator*(const Vec3& v, double t) {
    return Vec3(v.e[0]*t, v.e[1]*t, v.e[2]*t);
}
Vec3 operator*(double t, const Vec3& v) {
    return v*t;
}

// Output code
std::ostream& operator<<(std::ostream& out, const Vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

// Get unit vector
Vec3 unit_vector(const Vec3& v) {
    return v*(1/v.norm());
}