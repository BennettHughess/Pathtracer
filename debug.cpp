#include <string>
#include <vector>
#include "vec4.h"
#include "metric.h"
#include <iostream>
#include "path.h"
#include <cmath>

Vec3 get_manual_spherical_velocity3(Vec3 spos, Vec3 cvel) {

    // manually computing velocity vector
    Vec3 cartpvec = CoordinateSystem3::Spherical_to_Cartesian(spos);
    cartpvec = cartpvec + 0.1*cvel;
    cartpvec = CoordinateSystem3::Cartesian_to_Spherical(cartpvec);
    Vec3 minus_spherical_position = -spos;
    cartpvec += minus_spherical_position;
    std::cout << "cartpvec: " << cartpvec*10 << '\n';

    return cartpvec*10;

}

double get_spherical_speed(Vec3 spos, Vec3 svel) {

    double speed_squared { svel[0]*svel[0] 
        + spos[0]*spos[0]*sin(spos[2])*sin(spos[2])*svel[1]*svel[1] 
        + spos[0]*spos[0]*svel[2]*svel[2] };

    return std::sqrt(speed_squared);

}

Vec3 get_spherical_tangent(Vec3 cpos, Vec3 cvel) {

    double r { cpos.norm() };

    double Tr { cpos[0]*cvel[0]/r + cpos[1]*cvel[1]/r + cpos[2]*cvel[2]/r };

    double Ttheta { (1/(r*r*std::sqrt(cpos[0]*cpos[0]+cpos[1]*cpos[1]))) 
        * (cpos[0]*cpos[2]*cvel[0] + cpos[1]*cpos[2]*cvel[1] - (cpos[0]*cpos[0]+cpos[1]*cpos[1])*cvel[2] ) };

    double Tphi { (1/(cpos[0]*cpos[0]+cpos[1]*cpos[1]))*(-cpos[1]*cvel[0]+cpos[0]*cvel[1]) };

   // Vec3 tangent { Tr, Ttheta, Tphi };

    return Vec3 { Tr, Ttheta, Tphi };

}


int main() {

    double pi {3.14159};

    // Make two metrics
    Metric spherical_metric { Metric::SphericalMinkowskiMetric};
    Metric cartesian_metric { Metric::CartesianMinkowskiMetric};

    // Cartesian 3-position and 3-velocity
    Vec3 cartesian_position3 {1,1,1};
    Vec3 cartesian_velocity3 {1,1,1};


    // Spherical position and 3-velocity
    // Vec3 spherical_position3 { std::sqrt(3), pi/4, pi/4 };
    Vec3 spherical_position3 { CoordinateSystem3::Cartesian_to_Spherical(cartesian_position3) };
    Vec3 spherical_velocity3 { CoordinateSystem3::CartesianVector_to_SphericalVector(cartesian_velocity3, spherical_position3) };
    Vec3 cartpvec { get_manual_spherical_velocity3(spherical_position3, cartesian_velocity3) };
    Vec3 svel { get_spherical_tangent(cartesian_position3, cartesian_velocity3) };

    std::cout << "speed of cartesian: " << std::sqrt(cartesian_velocity3[0]*cartesian_velocity3[0] 
        + cartesian_velocity3[1]*cartesian_velocity3[1] 
        + cartesian_velocity3[2]*cartesian_velocity3[2]) << '\n';
    std::cout << "speed of spherical: " << get_spherical_speed(spherical_position3, svel) << '\n';
    std::cout << "svel: " << svel << '\n';

    // Make 4-positions and 4-velocities
    Vec4 cartesian_position4 {0, cartesian_position3};
    Vec4 cartesian_velocity4 {convert_to_null(cartesian_velocity3, cartesian_position4, cartesian_metric)};

    Vec4 spherical_position4 {0, spherical_position3};
    //Vec4 spherical_velocity4 {convert_to_null(cartpvec, spherical_position4, spherical_metric)}; // cartpvec version
    //Vec4 spherical_velocity4 {convert_to_null(spherical_velocity3, spherical_position4, spherical_metric)};
    Vec4 spherical_velocity4 {convert_to_null(svel, spherical_position4, spherical_metric)};


    // Spherical and cartesian paths
    Path::Integrator integrator {Path::Euler};
    Path cartesian_path { cartesian_position4, cartesian_velocity4, integrator };
    Path spherical_path { spherical_position4, spherical_velocity4, integrator };

    std::cout << "cartesian initial path position: " << cartesian_path.get_position() << " velocity: " << cartesian_path.get_velocity() << '\n';
    std::cout << "spherical initial path position: " << spherical_path.get_position() << " velocity: " << spherical_path.get_velocity() << '\n';


    // Propagate paths
    double dlam {0.0001};
    double radius {0};
    while (radius < 100) {

        cartesian_path.propagate(dlam, cartesian_metric);
        radius = cartesian_path.get_position().get_vec3().norm();

    }

    radius = 0;
    while (radius < 100) {

        spherical_path.propagate(dlam, spherical_metric);
        radius = spherical_path.get_position()[1];

    }

    std::cout << "cartesian position: " << cartesian_path.get_position().get_vec3() << '\n';
    std::cout << "spherical position: " << spherical_path.get_position().get_vec3() << '\n';

    Vec3 spherical_final_pos {spherical_path.get_position().get_vec3()};
    Vec3 spherical_final_pos_in_cart {CoordinateSystem3::Spherical_to_Cartesian(spherical_final_pos)};

    Vec3 cartesian_final_pos {cartesian_path.get_position().get_vec3()};
    Vec3 cartesian_final_pos_in_sphere {CoordinateSystem3::Cartesian_to_Spherical(cartesian_final_pos)};

    std::cout << "cartesian position in spherical coords: " << cartesian_final_pos_in_sphere << '\n';
    std::cout << "spherical position in cartesian coords: " << spherical_final_pos_in_cart << '\n';

    double difference {(spherical_final_pos_in_cart - cartesian_path.get_position().get_vec3()).norm()};

    std::cout << "difference: " << difference << '\n';


    return 0;
}