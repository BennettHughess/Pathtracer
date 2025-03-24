#include "../include/scenario.h"
#include <cmath>

bool Scenario::collision_checker(Path& path) {

    double radius;
    bool inside_background;
    bool far_from_event_horizon;
    double rho_s;
    double rho;

    switch (type) {

        case SphericalMinkowski:
            // get radius
            radius = path.get_position()[1];

            // Collision happens when photon is outside background
            inside_background = radius < params.background_radius;

            return inside_background;
            break;

        case CartesianMinkowski:
            // get radius
            radius = path.get_position().get_vec3().norm();

            // Collision happens when photon is outside background
            inside_background = radius < params.background_radius;

            return inside_background;
            break;

        case Schwarzschild:
            // get radius
            radius = path.get_position()[1];

            // Collision happens when photon is outside background
            inside_background = radius < params.background_radius;

            // or close to event horizon
            far_from_event_horizon = radius > 2.01*params.black_hole_mass;

            return inside_background && far_from_event_horizon;
            break;

        case CartesianSchwarzschild:
            rho = path.get_position().get_vec3().norm();
            rho_s = 2.*params.black_hole_mass/4;
            radius = rho*pow((1 + rho_s/rho),2);

            // Collision happens when photon is outside background
            inside_background = radius < params.background_radius;

            // or close to event horizon
            far_from_event_horizon = rho > 1.01*rho_s;

            return inside_background && far_from_event_horizon;
            break;

        default:
            std::cerr << "Scenario Error: unknown scenario type. Failing to check collision." << std::endl;
            return false;
            break;

    }
    
};

Vec3 Scenario::get_pixel_pos(Path& path) {

    Vec3 pos;
    double rho;
    double rho_s;
    double radius;

    switch (type) {

        case SphericalMinkowski:
            pos = path.get_position().get_vec3();
            break;

        case CartesianMinkowski:
            pos = CoordinateSystem3::Cartesian_to_Spherical(
                path.get_position().get_vec3()
            );
            break;

        case Schwarzschild:
            pos = path.get_position().get_vec3();
            break;

        case CartesianSchwarzschild:
            pos = CoordinateSystem3::Cartesian_to_Spherical(
                path.get_position().get_vec3()
            );

            // we need to convert from this rho coordinate to a normal radial one
            rho = pos[0];
            rho_s = 2.*params.black_hole_mass/4.;
            radius = rho*pow((1 + rho_s/rho),2);

            pos = {radius, pos[1], pos[2]};
            break;

        default:
            std::cerr << "Scenario Error: unknown scenario type. Failing to get path pos." << std::endl;
            break;

    }

    return pos;

};