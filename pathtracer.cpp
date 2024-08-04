#include "pathtracer.h"

// Update the position to given one
Path& Path::update_position(Vec3& new_position) {
    position = new_position;
    return *this;
}

// Update direction to given one
Path& Path::update_velocity(Vec3& new_velocity) {
    velocity = new_velocity;
    return *this;
}

// Propagate path forward
Path& Path::propagate(double dt) {
    // Should implement some kind of acceleration vector later
    // This would be another input to propagate, `Vec3& accel`
    position = position+velocity*dt;
    return *this;
}