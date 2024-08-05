#pragma once
#include <vector>
#include "vec4.h"

// forward declare vec4 to use!
class Vec4;

class Metric {
    public:

        // This is passed to the constructor during initialization
        enum MetricType{
            MinkowskiMetric,            // assumes (t, x, y, z) coordinates
            SchwarzschildMetric         // assumes (t, r, theta, phi) coordinates
        };
    
    private:

        // Metric possesses a type, is used to construct the rest of the metric
        MetricType type;

        // If type is Schwarzschild, needs a mass
        double mass;
    
    public:

        // Constructor
        Metric(MetricType t, double m = 1) : type {t}, mass {m} {}

        // Get metric components at a point in spacetime
        std::vector<double> get_components(const Vec4& position);

        // Computes second derivative of position and returns as a 4-vector
        Vec4 get_acceleration(Vec4& position, Vec4& velocity);

};