#pragma once
#include <cmath>

// define class Vec3
class Vec3 {
    public:
        // define vectors as having an array of three doubles (the components)
        double e[3];

        // initialize vectors like this: Vec3 vector {1, 20, 3};
        Vec3(double x, double y, double z) : e{x, y, z} {}

        /*
            VECTOR OPERATIONS 	ʕっ•ᴥ•ʔっ
        */

        // Negate a vector
        Vec3 operator-() const { 
            return Vec3(-e[0], -e[1], -e[2]); 
        }

        // Access vector as an array
        double operator[](int i) const {
            // note: no exception handling for out-of-bounds indices
            return e[i];
        }

        // Add vector to existing vector
        Vec3& operator+=(Vec3& v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }

        // Multiply vector by scalar
        Vec3& operator*=(double t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }

        // Compute norm
        double norm() const {
            return std::sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
        }
};

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
Vec3 operator*(Vec3& v, double t) {
    return Vec3(v.e[0]*t, v.e[1]*t, v.e[2]*t);
}
Vec3 operator*(double t, Vec3& v) {
    return v*t;
}