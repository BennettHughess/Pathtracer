#include "path.h"
#include <functional>
#include "metric.h"
#include <iostream>

// Standard Euler integrator
void Path::euler_propagate(double dlam, Metric& metric) {

    Vec4 accel = metric.get_acceleration(get_position(), get_velocity());

    set_position(position + velocity*dlam);
    set_velocity(velocity + accel*dlam);
    
}

// Verlet implementation is from https://arxiv.org/pdf/0909.0708 and wikipedia
void Path::verlet_propagate(double dlam, Metric& metric) {

    // current acceleration
    Vec4 initial_acceleration = metric.get_acceleration(get_position(), get_velocity());

    // x^alpha_{n+1}
    Vec4 new_position { position + velocity*dlam + 0.5*initial_acceleration*dlam*dlam };
    set_position(new_position);

    // k^alpha_{n+1,p}
    Vec4 velocity_prime { velocity + initial_acceleration*dlam };

    // accel_{n+1}
    Vec4 new_acceleration { metric.get_acceleration(new_position, velocity_prime) };

    // k^alpha_{n+1}
    Vec4 new_velocity { velocity + 0.5*(initial_acceleration + new_acceleration)*dlam };
    set_velocity(new_velocity);

}

// Propagate path forward
void Path::propagate(double dlam, Metric& metric) {

    // std::cout << "path.cpp: propagated \n";

    switch (integrator) {

        case Euler:
            euler_propagate(dlam, metric);

        case Verlet:
            verlet_propagate(dlam, metric);

    }    

}

// Propagate path until condition is no longer met
void Path::loop_propagate(std::function<bool(Path&)> condition, double dlam, Metric& metric) {
    // evaluate condition at every loop

    while (condition(*this)) {
        propagate(dlam, metric);
    }

}