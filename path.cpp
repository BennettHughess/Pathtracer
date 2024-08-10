#include "path.h"
#include <functional>
#include "metric.h"
#include <iostream>
#include <cmath>

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
void Path::rk4_propagate(double dlam, Metric& metric) {

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

//https://lilith.fisica.ufmg.br/~dickman/transfers/comp/textos/DeVries_A_first_course_in_computational_physics.pdf
double Path::rkf45_propagate(double dlam, Metric& metric) {

    // proceed very similarly to rk45
    // let y1 = pos, y2 = vel
    Vec4 y1 { get_position() };
    Vec4 y2 { get_velocity() };

    // first step: k1, l1
    Vec4 l1 { y2 };
    Vec4 k1 { metric.get_acceleration(y1,y2) };

    // second step: k2, l2
    Vec4 l2 { y2 + (dlam/4)*l1 };
    Vec4 k2 { metric.get_acceleration(y1 + (dlam/4)*l1, y2 + (dlam/4)*k1) };

    // third step: k3, l3
    Vec4 l3 { y2 + dlam*((3.0/32)*l1 + (9.0/32)*l2) };
    Vec4 k3 { metric.get_acceleration(y1 + dlam*((3.0/32)*l1 + (9.0/32)*l2 ), 
        y2 + dlam*((3.0/32)*k1 + (9.0/32)*k2 )) };

    // fourth step: k4, l4
    Vec4 l4 { y2 + dlam*((1932.0/2197)*l1 - (7200.0/2197)*l2 + (7296.0/2197)*l3) };
    Vec4 k4 { metric.get_acceleration(y1 + dlam*((1932.0/2197)*l1 - (7200.0/2197)*l2 + (7296.0/2197)*l3), 
        y2 + dlam*((1932.0/2197)*k1 - (7200.0/2197)*k2 + (7296.0/2197)*k3) ) };

    // fifth step k5, l5
    Vec4 l5 { y2 + dlam*((439.0/216)*l1 - 8.0*l2 + (3680.0/513)*l3 - (845.0/4104)*l4) };
    Vec4 k5 { metric.get_acceleration(y1 + dlam*((439.0/216)*l1 - 8.0*l2 + (3680.0/513)*l3 - (845.0/4104)*l4), 
        y2 + dlam*((439.0/216)*k1 - 8.0*k2 + (3680.0/513)*k3 - (845.0/4104)*k4)) };

    // fifth step k6, l6
    Vec4 l6 { y2 + dlam*((-8.0/27)*l1 + 2.0*l2 - (3544.0/2565)*l3 + (1854.0/4104)*l4 - (11.0/40)*l5) };
    Vec4 k6 { metric.get_acceleration(y1 + dlam*((-8.0/27)*l1 + 2.0*l2 - (3544.0/2565)*l3 + (1854.0/4104)*l4 - (11.0/40)*l5), 
        y2 + dlam*((-8.0/27)*k1 + 2.0*k2 - (3544.0/2565)*k3 + (1854.0/4104)*k4 - (11.0/40)*k5)) };

    // compute y1, y2 to orders 4 and 5
    Vec4 y1_order4 { y1 + dlam*((25.0/216)*l1 + (1408.0/2565)*l3 + (2197.0/4104)*l4 - (1.0/5)*l5) };
    Vec4 y2_order4 { y2 + dlam*((25.0/216)*k1 + (1408.0/2565)*k3 + (2197.0/4104)*k4 - (1.0/5)*k5) };
    Vec4 y1_order5 { y1 + dlam*((16.0/135)*l1 + (6656.0/12825)*l3 + (28561.0/56430)*l4 - (9.0/50)*l5 + (2.0/55)*l6) };
    Vec4 y2_order5 { y2 + dlam*((16.0/135)*k1 + (6656.0/12825)*k3 + (28561.0/56430)*k4 - (9.0/50)*k5 + (2.0/55)*k6) };

    // compute error
    Vec4 y1_error { y1_order5 - y1_order4 };
    Vec4 y2_error { y2_order5 - y2_order4 };

    // find largest error
    double max_error {std::abs(y1_error[0])};
    for (int i {1}; i < 4; ++i) {
        max_error = std::max(std::abs(y1_error[i]), max_error);
    }
    for (int i {0}; i < 4; ++i) {
        max_error = std::max(std::abs(y2_error[i]), max_error);
    }

    // compute new step size
    double tolerance { 0.0001 };
    double best_dlam {
        0.9*dlam*std::pow(std::abs(dlam)*tolerance/(max_error), 1/4.0)
    };

    double max_dlam { 0.001 };
    double min_dlam { 0.000001 };

    double new_dlam {};
    if (best_dlam > max_dlam) {
        new_dlam = max_dlam;
    }
    else if (best_dlam < min_dlam) {
        new_dlam = min_dlam;
    }
    else {
        new_dlam = best_dlam;
    }

    /*

    if (dlam > new_dlam) {
        // do it again

        // first step: k1, l1
        l1 = y2;
        k1 = metric.get_acceleration(y1,y2);

        // second step: k2, l2
        l2 = y2 + (new_dlam/4)*l1;
        k2 = metric.get_acceleration(y1 + (new_dlam/4)*l1, y2 + (new_dlam/4)*k1);

        // third step: k3, l3
        l3 = y2 + new_dlam*((3.0/32)*l1 + (9.0/32)*l2);
        k3 = metric.get_acceleration(y1 + new_dlam*((3.0/32)*l1 + (9.0/32)*l2 ), 
            y2 + new_dlam*((3.0/32)*k1 + (9.0/32)*k2 ));

        // fourth step: k4, l4
        l4 = y2 + new_dlam*((1932.0/2197)*l1 - (7200.0/2197)*l2 + (7296.0/2197)*l3);
        k4 = metric.get_acceleration(y1 + new_dlam*((1932.0/2197)*l1 - (7200.0/2197)*l2 + (7296.0/2197)*l3), 
            y2 + new_dlam*((1932.0/2197)*k1 - (7200.0/2197)*k2 + (7296.0/2197)*k3) );

        // fifth step k5, l5
        l5 = y2 + new_dlam*((439.0/216)*l1 - 8.0*l2 + (3680.0/513)*l3 - (845.0/4104)*l4);
        k5 = metric.get_acceleration(y1 + new_dlam*((439.0/216)*l1 - 8.0*l2 + (3680.0/513)*l3 - (845.0/4104)*l4), 
            y2 + new_dlam*((439.0/216)*k1 - 8.0*k2 + (3680.0/513)*k3 - (845.0/4104)*k4));

        // fifth step k6, l6
        l6 = y2 + new_dlam*((-8.0/27)*l1 + 2.0*l2 - (3544.0/2565)*l3 + (1854.0/4104)*l4 - (11.0/40)*l5);
        k6 = metric.get_acceleration(y1 + new_dlam*((-8.0/27)*l1 + 2.0*l2 - (3544.0/2565)*l3 + (1854.0/4104)*l4 - (11.0/40)*l5), 
            y2 + new_dlam*((-8.0/27)*k1 + 2.0*k2 - (3544.0/2565)*k3 + (1854.0/4104)*k4 - (11.0/40)*k5));

        // compute y1, y2 to orders 4 and 5
        y1_order4 = y1 + new_dlam*((25.0/216)*l1 + (1408.0/2565)*l3 + (2197.0/4104)*l4 - (1.0/5)*l5);
        y2_order4 = y2 + new_dlam*((25.0/216)*k1 + (1408.0/2565)*k3 + (2197.0/4104)*k4 - (1.0/5)*k5);
        y1_order5 = y1 + new_dlam*((16.0/135)*l1 + (6656.0/12825)*l3 + (28561.0/56430)*l4 - (9.0/50)*l5 + (2.0/55)*l6);
        y2_order5 = y2 + new_dlam*((16.0/135)*k1 + (6656.0/12825)*k3 + (28561.0/56430)*k4 - (9.0/50)*k5 + (2.0/55)*k6);

        //std::cout << "recalculated with " << new_dlam << '\n';
    }

    */

    // update position, velocity
    set_position(y1_order4);
    set_velocity(y2_order4);

    return new_dlam;

}

// from https://arxiv.org/pdf/0909.0708
void Path::refined_verlet_propagate(double dlam, Metric& metric) {

    // x0 is current pos, x1 is next pos
    // v0 is current vel, v1 is next vel
    // a0 is current accel, a0 is next accel
    Vec4 x0 {get_position()};
    Vec4 v0 {get_velocity()};
    Vec4 a0 {metric.get_acceleration(x0, v0)};

    // compute next position
    Vec4 x1 {x0 + v0*dlam + 0.5*a0*dlam*dlam};

    // estimate next velocity
    Vec4 v1prime {v0 + a0*dlam};

    // compute next acceleration
    Vec4 a1 {metric.get_acceleration(x1, v1prime)};

    // compute better velocity v1
    Vec4 v1 {v0 + 0.5*(a0 + a1)*dlam};

    // get fractional change in velocities:
    Vec4 reciprocal_v1 {1/v1[0], 1/v1[1], 1/v1[2], 1/v1[3]};
    double fracchange {std::abs(((v1 - v1prime)*reciprocal_v1).max())};

    // loop until fracchange is small (up to 5 times)
    int counter {0};
    while (fracchange > 0.001) {
        a1 = metric.get_acceleration(x1, v1);
        v1prime = v1;
        v1 = v1 + 0.5*(a0 + a1)*dlam;

        reciprocal_v1 = {1/v1[0], 1/v1[1], 1/v1[2], 1/v1[3]};
        fracchange = std::abs(((v1 - v1prime)*reciprocal_v1).max());

        if (counter > 4) {
            // std::cout << "had to break loop. fracchange: " << fracchange << '\n';
            break;
        }
        ++counter;
    }  

    //std::cout << "number of counters before breaking: " << counter << '\n';

    // set new position, velocity
    set_position(x1);
    set_velocity(v1);

}

// Propagate path forward
double Path::propagate(double dlam, Metric& metric) {

    // std::cout << "path.cpp: propagated \n";

    switch (integrator) {

        case Euler:
            euler_propagate(dlam, metric);
            break;

        case Verlet:
            verlet_propagate(dlam, metric);
            break;

        case RK4:
            rk4_propagate(dlam, metric);
            break;

        case RKF45:
            dlam = rkf45_propagate(dlam, metric);
            // std::cout << "position: " << get_position() << " dlam: " << dlam << '\n';
            break;

        case RefinedVerlet:
            refined_verlet_propagate(dlam, metric);
            break;

    }    
    return dlam;

}

// Propagate path until condition is no longer met
void Path::loop_propagate(std::function<bool(Path&)> condition, double dlam, Metric& metric) {
    // evaluate condition at every loop

    while (condition(*this)) {
        dlam = propagate(dlam, metric);
    }

}