#pragma once
#include <iostream>
#include "vec3.h"
#include "metric.h"

// forward declare metric to use!
class Metric;

// define class Vec4 (three spatial one time)
class Vec4 {
    private:

        // define 4-vectors as four doubles: e[0] is time, e[1] to e[3] are spatial
        double e[4];
        
    public:

        // Default constructor
        Vec4() : e {0,0,0,0} {}

        // Construct from an array
        Vec4(double v[4]) : e {v[0], v[1], v[2], v[3]} {}

        // Construct from four doubles
        Vec4(double e0, double e1, double e2, double e3) : e {e0, e1, e2, e3} {}

        // Construct from Vec3 and a time component
        Vec4(double v0, const Vec3& v) : e {v0, v[0], v[1], v[2]} {}

        // Access functions
        Vec3 get_vec3() { return Vec3{e[1],e[2],e[3]}; }

        // Operations on 4-vectors
        Vec4 operator-() const;

        double operator[](int i) const;

        Vec4& operator+=(Vec4& v);

        Vec4& operator*=(double t);

        // Note: norm depends on the metric, which depends on the position of the vector
        double norm(Metric& metric, Vec4& position);
};

// More forward declarations
Vec4 operator+(const Vec4& v1, const Vec4& v2);

// Subtract two vectors and store as new vector
Vec4 operator-(const Vec4& v1, const Vec4& v2);

// Multiply vector by scalar and save as new vector (made commutative)
Vec4 operator*(const Vec4& v, double t);
Vec4 operator*(double t, const Vec4& v);

// stream output code
std::ostream& operator<<(std::ostream& out, const Vec4& v);

// Turn any 3-vector into null 4-vector in the same direction
Vec4 convert_to_null(const Vec3& v, const Vec4& position, Metric& metric);

// take inner product of v1 and v2 given a position-dependent metric
double inner(const Vec4& v1, const Vec4& v2, Vec4& position, Metric& metric);