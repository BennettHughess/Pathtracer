#include "pathtracer.h"

// Update the position to given one
Path& Path::update_pos(Vec3& new_pos) {
    pos = new_pos;
    return *this;
}

// Update direction to given one
Path& Path::update_vel(Vec3& new_vel) {
    vel = new_vel;
    return *this;
}

// Propagate path forward
Path& Path::propagate(double dt) {
    // Should implement some kind of acceleration vector later
    // This would be another input to propagate, `Vec3& accel`
    pos = pos+vel*dt;
    return *this;
}