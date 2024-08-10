#pragma once
#include "vec4.h"
#include <functional>
#include "metric.h"

// Define class of paths
class Path {
    public:
        
        enum Integrator {
            Euler,
            Verlet,
            RK4
        };

    private:
        // Paths have a position and direction (4-vectors)
        Vec4 position;
        Vec4 velocity;

        // Paths need an integrator
        Integrator integrator;

        // Different types of integrators:

        // Typical euler integration
        void euler_propagate(double dlam, Metric& metric);

        // Velocity verlet integration https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
        void verlet_propagate(double dlam, Metric& metric);

        // RK4 algorithm https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#The_Runge%E2%80%93Kutta_method
        void RK4_propagate(double dlam, Metric& metric);
        

    public:

        // Constructors
        Path() : position {0,0,0,0}, velocity {1,0,0,0}, integrator {Euler} {}
        Path(const Vec4& pos, const Vec4& vel, Integrator integrator = Euler) : position {pos}, velocity {vel}, integrator {Euler} {}

        // Access functions
        Vec4& get_position() {return position;}
        Vec4& get_velocity() {return velocity;}

        void set_position(const Vec4& pos) {position = pos;}
        void set_velocity(const Vec4& vel) {velocity = vel;}
        void set_integrator(Integrator integ) {integrator = integ;}

        // Propagate path. dtau is the step size of the affine parameter
        void propagate(double dlam, Metric& metric);

        // Propagate path until condition (which is a lambda function) is met
        void loop_propagate(std::function<bool(Path&)> condition, double dlam, Metric& metric);

};