#include "../include/cuda_misc.cuh"

__device__ bool cuda_collision_checker(CudaPath& path, Scenario::ScenarioType& scenario_type, ScenarioParameters& params) {

    double radius;
    bool inside_background;
    bool far_from_event_horizon;
    double rho_s;
    double rho;

    switch (scenario_type) {

        case Scenario::SphericalMinkowski:
            // get radius
            radius = path.get_position()[1];

            // Collision happens when photon is outside background
            inside_background = radius < params.background_radius;

            return inside_background;
            break;

        case Scenario::CartesianMinkowski:
            // get radius
            radius = path.get_position().get_vec3().norm();

            // Collision happens when photon is outside background
            inside_background = radius < params.background_radius;

            return inside_background;
            break;

        case Scenario::Schwarzschild:
            // get radius
            radius = path.get_position()[1];

            // Collision happens when photon is outside background
            inside_background = radius < params.background_radius;

            // or close to event horizon
            far_from_event_horizon = radius > 2.01*params.black_hole_mass;

            return inside_background && far_from_event_horizon;
            break;

        case Scenario::CartesianSchwarzschild:
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
            printf("Scenario Error: unknown scenario type. Failing to check collision.");
            return false;
            break;

    }
    
}