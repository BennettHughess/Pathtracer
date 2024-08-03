#pragma once
#include "vec3.h"

// Define class of paths
class Path {
    public:
        // Paths have a position and direction (3-vectors)
        Vec3 pos;
        Vec3 vel;

        // Constructors
        Path() : pos {0,0,0}, vel {0,0,0} {}
        Path(const Vec3& position, const Vec3& velocity) : pos {position}, vel {velocity} {}

        // Update position
        Path& update_pos(Vec3& new_pos);

        // Update velocity
        Path& update_vel(Vec3& new_vel);

        // Propagate path
        Path& propagate(double dt);

};