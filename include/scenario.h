#pragma once
#include "metric.h"
#include "path.h"
#include "vec3.h"
#include <iostream>

// Forward declare path to avoid circular compiler errors
class Path;

// This struct contains the various parameters for the scenario, and is passed during construction
struct ScenarioParameters{

    double black_hole_mass {1};
    double background_radius {10};

};

class Scenario {

    public:

        // Each possible type of scenario belongs to this enum
        enum ScenarioType{
            SphericalMinkowski,         // Minkowski space in spherical coordinates
            CartesianMinkowski,         // Minkowski space in Cartesian coordinates
            Schwarzschild,              // Schwarzschild black hole in Schwarzschild coordinates (spherical)
            CartesianSchwarzschild      // Schwarzschild black hole in isotropic Cartesian coordinates
        };

    private:

        // Every scenario posseses a metric, scenario type, parameters for scenario
        Metric metric;
        ScenarioType type;
        ScenarioParameters params;

    public:

    // Constructors
    Scenario() {

        metric = Metric();
        type = CartesianMinkowski;

    };

    Scenario(ScenarioType scenario_type, ScenarioParameters scenario_params) {

        type = scenario_type;
        params = scenario_params;
        
        // initialize metric_params to pass to metric during initialization
        MetricParameters metric_params {
            scenario_params.black_hole_mass,
        };

        switch (type) {

            case SphericalMinkowski:
                metric = Metric(Metric::SphericalMinkowskiMetric, metric_params);
                break;

            case CartesianMinkowski:
                metric = Metric(Metric::CartesianMinkowskiMetric, metric_params);
                break;

            case Schwarzschild:
                metric = Metric(Metric::SchwarzschildMetric, metric_params);
                break;

            case CartesianSchwarzschild:
                metric = Metric(Metric::CartesianIsotropicSchwarzschildMetric, metric_params);
                break;

            default:
                std::cerr << "Scenario Error: unknown scenario type " << scenario_type << ". Defaulting to Cartesian Minkowski." << std::endl;
                metric = Metric(Metric::CartesianMinkowskiMetric, metric_params);
                break;

        }
    };

    // Check for collision to see if path should terminate
    bool collision_checker(Path& path);

    // Once path terminates, return final path position in global spherical coordinates to get pixel color later
    Vec3 get_pixel_pos(Path& path);

    // Access functions
    Metric get_metric() {return metric;}               
    ScenarioParameters get_params() {return params;}
    ScenarioType get_type() {return type;}

};