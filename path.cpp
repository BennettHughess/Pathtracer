#include "path.h"
#include <functional>
#include "metric.h"
#include <iostream>

// Propagate path forward
void Path::propagate(double dtau, Metric& metric) {

    Vec4 accel = metric.get_acceleration(get_position(), get_velocity());

    set_position(position + velocity*dtau);
    set_velocity(velocity + accel*dtau);

}

int i = 0;

// Propagate path until condition is no longer met
void Path::loop_propagate(std::function<bool(Path&)> condition, double dtau, Metric& metric) {
    // evaluate condition at every loop

    if (i == 0) {
    while (condition(*this)) {
        propagate(dtau, metric);
        std::clog << "i am ray 0,0 and i am at " << position << '\n';
    }
    }
    else {
    while (condition(*this)) {
        propagate(dtau, metric);
    }
    }
    ++i;
}