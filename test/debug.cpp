#include <string>
#include <vector>
#include "../include/vec4.h"
#include "../include/metric.h"
#include <iostream>
#include "../include/path.h"
#include "../include/background.h"
#include <cmath>


int main() {

    Metric metric { Metric::CartesianIsotropicSchwarzschildMetric, 0};

    // Cartesian 3-position and 3-velocity
    Vec3 cartesian_position3 {3,0,0};
    Vec3 cartesian_velocity3 {-1,0.000000000001,1};

    // Make 4-positions and 4-velocities
    Vec4 cartesian_position4 {0, cartesian_position3};
    Vec4 cartesian_velocity4 {convert_to_null(cartesian_velocity3, cartesian_position4, metric)};

    // Spherical and cartesian paths
    Path::Integrator integrator {Path::CashKarp};
    Path cartesian_path { cartesian_position4, cartesian_velocity4, integrator };

    std::cout << "cartesian initial path position: " << cartesian_path.get_position() 
        << " velocity: " << cartesian_path.get_velocity() << '\n';

    // Propagate paths
    double dlam {0.0001};
    double radius {0};
    double max_dlam {0.1};
    double min_dlam {1E-20};
    double tolerance {1E-9};

    cartesian_path.set_max_dlam(max_dlam);
    cartesian_path.set_min_dlam(min_dlam);
    cartesian_path.set_tolerance(tolerance);

    while (radius < 5) {

        std::cout << metric.get_acceleration(cartesian_path.get_position(),
            cartesian_path.get_velocity()) << std::endl;
        cartesian_path.propagate(dlam, metric);
        radius = cartesian_path.get_position().get_vec3().norm();
        

    }

    Vec3 cartesian_final_pos {cartesian_path.get_position().get_vec3()};
    Vec3 cartesian_final_pos_in_sphere {CoordinateSystem3::Cartesian_to_Spherical(cartesian_final_pos)};

    std::cout << "cartesian position in spherical coords: " << cartesian_final_pos_in_sphere << '\n';
    std::cout << "cartesian position in cart coords: " << cartesian_final_pos << '\n';

    double rho = cartesian_final_pos_in_sphere.norm();
    double rho_s = 2./4.;
    radius = rho*pow((1 + rho_s/rho),2);

    Vec3 pos = {radius, cartesian_final_pos_in_sphere[2], cartesian_final_pos_in_sphere[3]};

    Background background {10, Background::Type::Image};
    background.load_ppm("/Users/ben/Code/Pathtracer/images/milky_way_hres.ppm");
    Vec3 pixel_color = background.get_color(cartesian_final_pos_in_sphere);

    std::cout << pixel_color << std::endl;

    //double difference {(cartesian_final_pos - cartesian_path.get_position().get_vec3()).norm()};

    return 0;
}