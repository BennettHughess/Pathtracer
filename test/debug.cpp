#include <string>
#include <vector>
#include "vec4.h"
#include "metric.h"
#include <iostream>
#include "path.h"
#include <cmath>


int main() {

    // Make two metrics
    Metric spherical_metric { Metric::SphericalMinkowskiMetric};
    Metric cartesian_metric { Metric::CartesianMinkowskiMetric};

    // Cartesian 3-position and 3-velocity
    Vec3 cartesian_position3 {1,0,0};
    Vec3 cartesian_velocity3 {-1,0.000000000001,1};


    // Spherical position and 3-velocity
    Vec3 spherical_position3 { CoordinateSystem3::Cartesian_to_Spherical(cartesian_position3) };
    Vec3 spherical_velocity3 { CoordinateSystem3::CartesianTangent_to_SphericalTangent(cartesian_position3, cartesian_velocity3) };

    // Make 4-positions and 4-velocities
    Vec4 cartesian_position4 {0, cartesian_position3};
    Vec4 cartesian_velocity4 {convert_to_null(cartesian_velocity3, cartesian_position4, cartesian_metric)};

    Vec4 spherical_position4 {0, spherical_position3};
    Vec4 spherical_velocity4 {convert_to_null(spherical_velocity3, spherical_position4, spherical_metric)};


    // Spherical and cartesian paths
    Path::Integrator integrator {Path::CashKarp};
    Path cartesian_path { cartesian_position4, cartesian_velocity4, integrator };
    Path spherical_path { spherical_position4, spherical_velocity4, integrator };

    std::cout << "cartesian initial path position: " << cartesian_path.get_position() << " velocity: " << cartesian_path.get_velocity() << '\n';
    std::cout << "spherical initial path position: " << spherical_path.get_position() << " velocity: " << spherical_path.get_velocity() << '\n';


    // Propagate paths
    double dlam {0.0001};
    double radius {0};
    double max_dlam {0.1};
    double min_dlam {1E-20};
    double tolerance {1E-9};

    spherical_path.set_max_dlam(max_dlam);
    spherical_path.set_min_dlam(min_dlam);
    spherical_path.set_tolerance(tolerance);
    cartesian_path.set_max_dlam(max_dlam);
    cartesian_path.set_min_dlam(min_dlam);
    cartesian_path.set_tolerance(tolerance);

    double cumulative_spherical_normsquared {
        spherical_path.get_velocity().norm_squared(spherical_metric, spherical_path.get_position())
    };
    while (radius < 2) {

        cartesian_path.propagate(dlam, cartesian_metric);
        radius = cartesian_path.get_position().get_vec3().norm();

    }

    radius = 0;
    double current_dlam {dlam};
    while (radius < 2) {

        // spherical_path.null_normalize(spherical_metric);
        current_dlam = spherical_path.propagate(current_dlam, spherical_metric);
        radius = spherical_path.get_position()[1];
        cumulative_spherical_normsquared += spherical_path.get_velocity().norm_squared(spherical_metric, spherical_path.get_position());
        //std::clog << "normsquared: " << spherical_path.get_velocity().norm_squared(spherical_metric, spherical_path.get_position()) << '\n';
        //std::clog << spherical_path.get_velocity() << '\n';
        //std::clog << spherical_path.get_position() << '\n';
        std::clog << "current dlam: " << current_dlam << '\n';
        

    }

    Vec3 spherical_final_pos {spherical_path.get_position().get_vec3()};
    Vec3 spherical_final_pos_in_cart {CoordinateSystem3::Spherical_to_Cartesian(spherical_final_pos)};

    Vec3 cartesian_final_pos {cartesian_path.get_position().get_vec3()};
    Vec3 cartesian_final_pos_in_sphere {CoordinateSystem3::Cartesian_to_Spherical(cartesian_final_pos)};

    std::cout << "cartesian position in spherical coords: " << cartesian_final_pos_in_sphere << '\n';
    std::cout << "spherical position in spherical coords: " << spherical_final_pos << '\n';

    double difference {(spherical_final_pos_in_cart - cartesian_path.get_position().get_vec3()).norm()};

    std::cout << "difference: " << difference << '\n';
    std::cout << "spherical cumulative norm^2: " << cumulative_spherical_normsquared << '\n';


    return 0;
}