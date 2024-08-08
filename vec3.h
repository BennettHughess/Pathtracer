#pragma once
#include <iostream>

// define class Vec3
class Vec3 {
    private:
    
        // define vectors as having an array of three doubles (the components)
        double e[3];
    
    public:
        // Constructors
        Vec3() : e{0,0,0} {}
        Vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}
        Vec3(double v[3]) : e { v[0], v[1], v[2] } {}
        
        // Copy constructor
        Vec3(const Vec3& v) : e{v[0], v[1], v[2]} {}

        //Function forward declarations
        Vec3 operator-() const;

        double operator[](int i) const;

        Vec3& operator+=(Vec3& v);

        Vec3& operator*=(double t);

        double norm() const;
};

// More forward declarations
Vec3 operator+(const Vec3& v1, const Vec3& v2);

// Subtract two vectors and store as new vector
Vec3 operator-(const Vec3& v1, const Vec3& v2);

// Take dot product of two vectors and store as double
double dot(const Vec3& v1, const Vec3& v2);

// Multiply vector by scalar and save as new vector (made commutative)
Vec3 operator*(const Vec3& v, double t);
Vec3 operator*(double t, const Vec3& v);

// stream output code
std::ostream& operator<<(std::ostream& out, const Vec3& v);

// Get unit vector
Vec3 unit_vector(const Vec3& v);

// Get cross product
Vec3 cross(const Vec3& v1, const Vec3& v2);

/*
    COORDINATE SYSTEMS
*/


// 3-vectors have a coordinate system
namespace CoordinateSystem3 {

    Vec3 Cartesian_to_Spherical(Vec3& cartesian);

    Vec3 Spherical_to_Cartesian(Vec3& spherical);

    // converts a cartesian vector at a given spherical position to a spherical vector
    Vec3 CartesianVector_to_SphericalVector(Vec3& cartesian_vec, Vec3& spherical_pos);

    // convert spherical position into the correct domains
    Vec3 Normalize_SphericalVector(Vec3& spherical);
    
};