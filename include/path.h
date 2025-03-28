#pragma once
#include "vec4.h"
#include <functional>
#include "metric.h"
#include "scenario.h"

// Forward declare scenario to avoid circular dependencies
class Scenario;

// Define class of paths
class Path {
    public:
        
        enum Integrator {
            Euler,
            Verlet,
            RK4,
            RKF45,
            CashKarp
        };

    private:
        // Paths have a position and direction (4-vectors)
        Vec4 position;
        Vec4 velocity;

        // Paths need an integrator to integrate the geodesic equations
        Integrator integrator;

        // for adaptive integrators, we need a minimum step size, maximum step size, and tolerance
        double tolerance {0.000001};
        double min_dlam {0.000001};
        double max_dlam {0.1};

        // Different types of integrators:

        // Typical euler integration
        void euler_propagate(double dlam, Metric& metric);

        // Velocity verlet integration https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
        void verlet_propagate(double dlam, Metric& metric);

        // RK4 algorithm https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#The_Runge%E2%80%93Kutta_method
        void rk4_propagate(double dlam, Metric& metric);

        // rkf45 has an auxiliary function to make the code nicer
        double rkf45_propagate(double dlam, Metric& metric);
        std::vector<Vec4> rkf45_integrate(double dlam, Metric& metric);

        //cashkarp
        double cashkarp_propagate(double dlam, Metric& metric);
        std::vector<Vec4> cashkarp_integrate(double dlam, Metric& metric);
        
    public:

        // Constructors
        Path() : position {0,0,0,0}, velocity {1,0,0,0}, integrator {Euler} {}
        Path(const Vec4& pos, const Vec4& vel, Integrator integ = Euler) : position {pos}, velocity {vel}, integrator {integ} {}

        // Access functions
        Vec4& get_position() {return position;}
        Vec4& get_velocity() {return velocity;}
        
        Integrator get_integrator() {return integrator;}
        double get_tolerance() {return tolerance;}
        double get_min_dlam() {return min_dlam;}
        double get_max_dlam() {return max_dlam;}

        void set_position(const Vec4& pos) {position = pos;}
        void set_velocity(const Vec4& vel) {velocity = vel;}

        void set_integrator(Integrator integ) {integrator = integ;}
        void set_min_dlam(double min) {min_dlam = min;}
        void set_max_dlam(double max) {max_dlam = max;}
        void set_tolerance(double tol) {tolerance = tol;}

        // Propagate path. dlam is the step size of the affine parameter
        // Returns a dlam (for use with adaptive step sizes)
        double propagate(double dlam, Metric& metric);

        // Propagate path until condition is met
        void loop_propagate(Scenario& scenario, double dlam);

        // Renormalize time component of photon velocity so that it is a null vector
        void null_normalize(Metric& metric);

};