#pragma once
#include "vec3.h"

// Define class of paths
class Path {
    public:
        // Paths have a position and direction (3-vectors)
        Vec3 position;
        Vec3 velocity;

        // Constructors
        Path() : position {0,0,0}, velocity {0,0,0} {}
        Path(const Vec3& pos, const Vec3& vel) : position {pos}, velocity {vel} {}

        // Update position
        Path& update_position(Vec3& new_position);

        // Update velocity
        Path& update_velocity(Vec3& new_velocity);

        // Propagate path
        Path& propagate(double dt);

};