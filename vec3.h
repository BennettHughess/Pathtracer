#pragma once

// define class Vec3
class Vec3 {
    public:
        // define vectors as having an array of three doubles (the components)
        double e[3];

        // Constructors
        Vec3() : e{0,0,0} {}
        Vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

        //Function forward declarations
        Vec3 operator-() const;

        double operator[](int i) const;

        Vec3& operator+=(Vec3& v);

        Vec3& operator*=(double t);

        double norm() const;
};