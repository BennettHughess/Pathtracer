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
    return Vec3(v1[0]+v2[0], v1[1]+v2[1], v1[2]+v2[2]);
}

// Subtract two vectors and store as new vector
Vec3 operator-(const Vec3& v1, const Vec3& v2) {
    return v1 + (-v2);
}

// Take dot product of two vectors and store as double
double dot(const Vec3& v1, const Vec3& v2) {
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]; 
}

// Multiply vector by scalar and save as new vector (made commutative)
Vec3 operator*(const Vec3& v, double t) {
    return Vec3(v[0]*t, v[1]*t, v[2]*t);
}
Vec3 operator*(double t, const Vec3& v) {
    return v*t;
}

// Output code
std::ostream& operator<<(std::ostream& out, const Vec3& v) {
    return out << v[0] << ' ' << v[1] << ' ' << v[2];
}

// Get unit vector
Vec3 unit_vector(const Vec3& v) {
    return v*(1/v.norm());
}

// Get cross product
Vec3 cross(const Vec3& v1, const Vec3& v2) {
    return Vec3{
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    };
}

/*
    COORDINATE SYSTEMS
*/

double m_pi = 3.14159;

Vec3 CoordinateSystem3::Cartesian_to_Spherical(Vec3& cartesian) {
    
    double r { cartesian.norm() };
    double theta {};
    double phi {};

    if (r != 0) {

        theta = atan2(sqrt(cartesian[0]*cartesian[0]+cartesian[1]*cartesian[1]),cartesian[2]);
        
        /*
        if (cartesian[0] >= 0 && cartesian[1] >= 0) {
            phi = atan(cartesian[1]/cartesian[0]);
        }
        else if (cartesian[0] < 0 && cartesian[1] >= 0) {
            phi = m_pi/2 + atan(-cartesian[0]/cartesian[1]);
        }
        else if (cartesian[0] < 0 && cartesian[1] < 0) {
            phi = m_pi + atan(cartesian[1]/cartesian[0]);
        }
        else if (cartesian[0] >= 0 && cartesian[1] < 0) {
            phi = 3*m_pi/2 + atan(-cartesian[0]/cartesian[1]);
        }
        else {
            std::clog << "Issue converting cartesian to spherical." << '\n';
        }
        */
        
        phi = atan2(cartesian[1],cartesian[0]);

    }
    else {
        theta = 0;
        phi = 0;
    }
    return Vec3 {r, theta, phi};
}


Vec3 CoordinateSystem3::Spherical_to_Cartesian(Vec3& spherical) {

    double x { spherical[0]*sin(spherical[1])*cos(spherical[2]) };
    double y { spherical[0]*sin(spherical[1])*sin(spherical[2]) };
    double z { spherical[0]*cos(spherical[1]) };

    return Vec3 {x, y, z};
}


// cvec is  a cartesian vector, spos is spherical position
Vec3 CoordinateSystem3::CartesianVector_to_SphericalVector(Vec3& cvec, Vec3& spos) {

    double r {  
       cvec[0]*cos(spos[2])*sin(spos[1]) + cvec[1]*sin(spos[1])*sin(spos[2]) + cvec[2]*cos(spos[1])
    };
    double theta {
        cvec[0]*cos(spos[1])*cos(spos[2]) + cvec[1]*cos(spos[1])*sin(spos[2]) - cvec[2]*sin(spos[1])
    };
    double phi { 
        -cvec[0]*sin(spos[2]) + cvec[1]*cos(spos[2])
    };

    return Vec3 {r, theta, phi};

}


// Maps vector to itself
Vec3 IdentityMap(Vec3& v) {
    return v;
}
