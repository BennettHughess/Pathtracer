#include "../include/path.h"
#include <functional>
#include "../include/metric.h"
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

    // do the rkf45 integration step
    // returns a list: {y1_order4, y1_order5, y2_order4, y2_order5}
    std::vector<Vec4> ylist { rkf45_integrate(dlam, metric) };

    // compute error: difference in the order 5 and order 4 solutions
    Vec4 y1_error { ylist[1] - ylist[0] };
    Vec4 y2_error { ylist[3] - ylist[2] };

    // find largest error
    double max_error {std::abs(y1_error[0])};
    for (int i {1}; i < 4; ++i) {
        max_error = std::max(std::abs(y1_error[i]), max_error);
    }
    for (int i {0}; i < 4; ++i) {
        max_error = std::max(std::abs(y2_error[i]), max_error);
    }

    // compute new step size
    double best_dlam {
        0.9*dlam*std::pow(std::abs(dlam)*tolerance/(max_error), 1/4.0)
    };

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


    if (dlam > new_dlam) {
        // reintegrate and use the new values
        ylist = rkf45_integrate(new_dlam, metric);
    }

    // update position, velocity
    set_position(ylist[0]);
    set_velocity(ylist[2]);

    return new_dlam;

}

// do one step of rkf45 integration
std::vector<Vec4> Path::rkf45_integrate(double dlam, Metric& metric) {

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

    return std::vector<Vec4> { y1_order4, y1_order5, y2_order4, y2_order5 };

}

// cash karp!!
double Path::cashkarp_propagate(double dlam, Metric& metric) {

    // do the cashkarp integration step
    // returns a list: {y1_order4, y1_order5, y2_order4, y2_order5}
    std::vector<Vec4> ylist { cashkarp_integrate(dlam, metric) };

    // compute error: difference in the order 5 and order 4 solutions
    Vec4 y1_error { ylist[1] - ylist[0] };
    Vec4 y2_error { ylist[3] - ylist[2] };
    std::vector<double> err_list {y1_error[0], y1_error[1], y1_error[2], y1_error[3],
        y2_error[0], y2_error[1], y2_error[2], y2_error[3]};

    // find largest error and record which one it is
    double max_error {std::abs(err_list[0])};
    int index_error {0};
    for (int i {1}; i < 8; ++i) {
        if (std::abs(err_list[i]) >= max_error) {
            max_error = std::abs(err_list[i]);
            index_error = i;
        }
    }

    // get the new y value with the most error in it
    double y_max_error {0};
    if (index_error < 4) {
        y_max_error = std::abs(ylist[1][index_error]);
    }
    else {
        y_max_error = std::abs(ylist[3][index_error-4]);
    }

    // compute new step size
    // note: i think its better if the acceptable error is *fractional*, i.e. error = tolerance*y
    double best_dlam {
        0.9*dlam*std::pow( tolerance*y_max_error/max_error , 1.0/5)
    };

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


    if (dlam > new_dlam) {
        // reintegrate and use the new values
        ylist = cashkarp_integrate(new_dlam, metric);
    }

    // update position, velocity with the fifth order estimate
    set_position(ylist[1]);
    set_velocity(ylist[3]);

    return new_dlam;

}

// do one step of cashkarp integration
std::vector<Vec4> Path::cashkarp_integrate(double dlam, Metric& metric) {

    // proceed very similarly to rk45
    // let y1 = pos, y2 = vel
    Vec4 y1 { get_position() };
    Vec4 y2 { get_velocity() };

    // first step: k1, l1
    Vec4 l1 { dlam*y2 };
    Vec4 k1 { dlam*metric.get_acceleration(y1,y2) };

    // k2, l2
    Vec4 l2 { dlam*(y2 + (1.0/5)*l1) };
    Vec4 k2 { dlam*metric.get_acceleration(y1 + (1.0/5)*l1, y2 + (1.0/5)*k1) };

    // k3, l3
    Vec4 l3 { dlam*(y2 + (3.0/40)*l1 + (9.0/40)*l2) };
    Vec4 k3 { dlam*metric.get_acceleration(y1 + (3.0/40)*l1 + (9.0/40)*l2, y2 + (3.0/40)*k1 + (9.0/40)*k2) };

    // k4, l4
    Vec4 l4 { dlam*(y2 + (3.0/10)*l1 - (9.0/10)*l2 + (6.0/5)*l3) };
    Vec4 k4 { dlam*metric.get_acceleration(y1 + (3.0/10)*l1 - (9.0/10)*l2 + (6.0/5)*l3,
        y2 + (3.0/10)*k1 - (9.0/10)*k2 + (6.0/5)*k3) };

    // k5, l5
    Vec4 l5 { dlam*(y2 - (11.0/54)*l1 + (5.0/2)*l2 - (70.0/27)*l3 + (35.0/27)*l4) };
    Vec4 k5 { dlam*metric.get_acceleration(y1 - (11.0/54)*l1 + (5.0/2)*l2 - (70.0/27)*l3 + (35.0/27)*l4,
        y2 - (11.0/54)*k1 + (5.0/2)*k2 - (70.0/27)*k3 + (35.0/27)*k4) };

    // k6, l6
    Vec4 l6 { dlam*(y2 + (1631.0/55296)*l1 + (175.0/512)*l2 + (575.0/13824)*l3 + (44275.0/110592)*l4 + (253.0/4096)*l5) };
    Vec4 k6 { dlam*metric.get_acceleration(y1 + (1631.0/55296)*l1 + (175.0/512)*l2 + (575.0/13824)*l3 + (44275.0/110592)*l4 + (253.0/4096)*l5,
        y2 + (1631.0/55296)*k1 + (175.0/512)*k2 + (575.0/13824)*k3 + (44275.0/110592)*k4 + (253.0/4096)*k5) };

    // compute y1, y2
    Vec4 y1_order4 { y1 + (2825.0/27648)*l1 + (18575.0/48384)*l3 + (13525.0/55296)*l4 + (277.0/14336)*l5 + (1.0/4)*l6 };
    Vec4 y2_order4 { y2 + (2825.0/27648)*k1 + (18575.0/48384)*k3 + (13525.0/55296)*k4 + (277.0/14336)*k5 + (1.0/4)*k6 };
    Vec4 y1_order5 { y1 + (37.0/378)*l1 + (250.0/621)*l3 + (125.0/594)*l4 + (512.0/1771)*l6 };
    Vec4 y2_order5 { y2 + (37.0/378)*k1 + (250.0/621)*k3 + (125.0/594)*k4 + (512.0/1771)*k6 };

    return std::vector<Vec4> { y1_order4, y1_order5, y2_order4, y2_order5 };

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
            break;

        case CashKarp:
            dlam = cashkarp_propagate(dlam, metric);
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

// Renormalize time component of photon velocity so that it is a null vector
void Path::null_normalize(Metric& metric) {

    std::vector<double> met = metric.get_components(position);

    // for null vector with diagonal metric: v0^2 = (-1/g_00) v^i v_i
    double v0 { 
        sqrt((-1/met[0])*(met[1]*velocity[0]*velocity[0] + met[2]*velocity[1]*velocity[1] + met[3]*velocity[2]*velocity[2])) 
    };

    Vec4 new_velocity {v0, velocity[1], velocity[2], velocity[3]};

    set_velocity(new_velocity);

}