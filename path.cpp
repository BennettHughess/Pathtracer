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

// RK4 propagation from wikipedia and stack exchange:
// https://math.stackexchange.com/questions/721076/help-with-using-the-runge-kutta-4th-order-method-on-a-system-of-2-first-order-od
void Path::RK4_propagate(double dlam, Metric& metric) {

    // we have two systems of equations: x'' = accel, and x' = vel.
    // first we get the current position and velocities. let y1 = pos, y2 = vel.
    Vec4 y1 { get_position() };
    Vec4 y2 { get_velocity() };

    // first step: getting k1 (for the y2 vector) and l1 (for the y1 vector)
    Vec4 l1 { dlam*y2 };
    Vec4 k1 { dlam*metric.get_acceleration(y1, y2) };

    // second step: getting k2 and l2
    Vec4 l2 { dlam*(y2 + 0.5*l1) };
    Vec4 k2 { dlam*metric.get_acceleration(y1 + 0.5*l1, y2 + 0.5*k1) };

    // third step: getting k3 and l3
    Vec4 l3 { dlam*(y2 + 0.5*l2) };
    Vec4 k3 { dlam*metric.get_acceleration(y1 + 0.5*l2, y2 + 0.5*k2) };

    // fourth step: getting k4 and l4
    Vec4 l4 { dlam*(y2 + l3) };
    Vec4 k4 { dlam*metric.get_acceleration(y1 + l3, y2 + k3) };

    // fifth step: update position and velocity with new values
    set_position(y1 + (double(1)/double(6))*(l1 + 2*l2 + 2*l3 + l4));
    set_velocity(y2 + (double(1)/double(6))*(k1 + 2*k2 + 2*k3 + k4));

}

// Propagate path forward
void Path::propagate(double dlam, Metric& metric) {

    // std::cout << "path.cpp: propagated \n";

    switch (integrator) {

        case Euler:
            euler_propagate(dlam, metric);
            break;

        case Verlet:
            verlet_propagate(dlam, metric);
            break;

        case RK4:
            RK4_propagate(dlam, metric);
            break;

    }    

}

// Propagate path until condition is no longer met
void Path::loop_propagate(std::function<bool(Path&)> condition, double dlam, Metric& metric) {
    // evaluate condition at every loop

    while (condition(*this)) {
        propagate(dlam, metric);
    }

}