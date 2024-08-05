#pragma once
#include "vec4.h"
#include <functional>
#include "metric.h"

// Define class of paths
class Path {
    private:
        // Paths have a position and direction (4-vectors)
        Vec4 position;
        Vec4 velocity;

    public:

        // Constructors
        Path() : position {0,0,0,0}, velocity {1,0,0,0} {}
        Path(const Vec4& pos, const Vec4& vel) : position {pos}, velocity {vel} {}

        // Access functions
        Vec4& get_position() {return position;}
        Vec4& get_velocity() {return velocity;}

        void set_position(const Vec4& pos) {position = pos;}
        void set_velocity(const Vec4& vel) {velocity = vel;}

        // Propagate path. dtau is the step size of the affine parameter
        void propagate(double dtau, Metric& metric);

        // Propagate path until condition (which is a lambda function) is met
        void loop_propagate(std::function<bool(Path&)> condition, double dt, Metric& metric);

};