#include <string>
#include <vector>
#include "vec4.h"
#include "metric.h"
#include <iostream>
#include "path.h"

int main() {

    double pi {3.14159};

    Vec3 spherical_position { 1, pi/2, pi };

    Vec3 cartesian_vector { 0, 0, -1 };

    Vec3 new_vector { CoordinateSystem3::CartesianVector_to_SphericalVector(cartesian_vector, spherical_position) };

    std::cout << cartesian_vector << " at " << spherical_position << " is " << new_vector << '\n';

    /*
    double pi {3.14159};

    double black_hole_mass {10};

    Metric metric { Metric::CartesianMinkowskiMetric, black_hole_mass };

    Vec4 position {0, 0, 0, 0};

    Vec3 velocity3 {1, 0, 0};

    Vec4 null_velocity {convert_to_null(velocity3, position, metric)};

    Path path {position, null_velocity, Path::Verlet};

    // Define the "not colliding" conditions
    std::function<bool(Path&)> within_radius = [black_hole_mass, &metric](Path& path) -> bool {
        
        // get radius
        double radius = path.get_position().get_vec3().norm();
        std::cout << "positon: " << path.get_position() << " with norm " << path.get_velocity().norm_squared(metric, path.get_position()) << '\n';

        // Collision happens 
        bool not_collided_bool { radius < 100};

        return not_collided_bool;
    };

    std::cout << "initial velocity: " << path.get_velocity() << " with norm squared " 
        << path.get_velocity().norm_squared(metric, path.get_position()) << '\n';

    path.loop_propagate(within_radius, 0.001, metric);

    std::cout << "final velocity: " << path.get_velocity() << " with norm squared " 
        << path.get_velocity().norm_squared(metric, path.get_position()) << '\n';

    */

    return 0;
}