#pragma once
#include <vector>
#include "vec4.h"

// forward declare vec4 to use! (why do I need this)
class Vec4;

// This struct contains the various parameters for the metric, and is passed during construction
struct MetricParameters{

    double black_hole_mass {1};

};

class Metric {
    public:

        // This is passed to the constructor during initialization
        enum MetricType{
            SphericalMinkowskiMetric,                   // assumes (t, r, theta, phi) coordinates
            CartesianMinkowskiMetric,                   // assumes (t, x, y, z) coordinates
            SchwarzschildMetric,                        // assumes (t, r, theta, phi) coordinates
            CartesianIsotropicSchwarzschildMetric       // assumes (t, x, y, z) coordinates 
                                                            // (but with an implicit rho cordinate)
        };
    
    private:

        // Metric possesses a type, is used to construct the rest of the metric
        MetricType type;

        // Also needs parameters (for now, just mass)
        MetricParameters params;
    
    public:

        // Constructor
        Metric(MetricType t, MetricParameters p) : type {t}, params {p} {}
        Metric() : type {CartesianIsotropicSchwarzschildMetric}, params {} {}

        // Access functions
        MetricType get_type() { return type; }
        MetricParameters get_params() { return params; }

        // Get metric components at a point in spacetime
        std::vector<double> get_components(const Vec4& position);

        // Computes second derivative of position and returns as a 4-vector
        Vec4 get_acceleration(const Vec4& position, const Vec4& velocity);

};