#include "metric.h"
#include <cmath>

// Get components of metric given a position
std::vector<double> Metric::get_components(const Vec4& position) {

    std::vector<double> components;

    // Check which type of metric it is
    switch (type) {

        case MinkowskiMetric:

            components = {-1, 1, 1, 1};
            break;
            
        case SchwarzschildMetric:

            components = {
                -(1 - 2*mass/position[1]), 
                1/(1 - 2*mass/position[1]), 
                position[1]*position[1], 
                position[1]*position[1]*sin(position[2])*sin(position[2])
            };
            break;

    }
    return components;

}

// Computes second derivative of position and returns as a 4-vector
Vec4 Metric::get_acceleration(Vec4& pos, Vec4& vel) {

    Vec4 acceleration;

     switch (type) {

        case MinkowskiMetric:

            // All christoffel symbols are zero for the Minkowski metric in (t,x,y,z) coords
            acceleration = {
                0,
                0,
                0,
                0
            };
            break;
            
        case SchwarzschildMetric:

            // Acceleration is computed from the geodesic equation
            acceleration = {
                (-2*mass/(pos[1]*pos[1]))*(1/(1 - 2*mass/pos[1]))*vel[1]*vel[0],            // time component
                (-mass/(pos[1]*pos[1]))*(1 - 2*mass/pos[1])*vel[0]*vel[0]                   // this whole thing is
                    + (mass/(pos[1]*pos[1]))*(1/(1 - 2*mass/pos[1]))*vel[1]*vel[1]          // the radial component
                    + pos[1]*(1 - 2*mass/pos[1])*vel[2]*vel[2]
                    + pos[1]*sin(pos[2])*sin(pos[2])*(1 - 2*mass/pos[1])*vel[3]*vel[3],
                (-2/pos[1])*vel[2]*vel[1] + sin(pos[2])*cos(pos[2])*vel[2]*vel[2],          // theta component
                (-2/pos[1])*vel[3]*vel[1] - (2/tan(pos[2]))*vel[3]*vel[2]                   // phi component
            };
            break;

    }
    return acceleration;

}