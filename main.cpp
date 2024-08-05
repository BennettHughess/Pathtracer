#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <functional>
#include "vec3.h"
#include "vec4.h"
#include "camera.h"
#include "path.h"
#include "background.h"

int main(int argc, char *argv[]) {

    // Configure camera position and direction
    Vec3 camera_position {-5,0,0};
    Vec3 camera_direction {1,0,0};
    Vec3 camera_up {0,0,1};
    Camera camera {camera_position, camera_direction, camera_up};

    double pi = 3.141459;
    // Rotate camera!
    camera.rotate(0,0,0);

    // Configure image size
    const int image_width {1920};
    const int image_height {1080};
    camera.set_image_settings(image_width, image_height);

    // Get filename and initialize filestream
    std::ofstream filestream;
    if (argv[1] == NULL) {        // check if filename was inputted
        filestream.open("main.ppm"); // if not, default output file to main.ppm
    } 
    else {
        std::string filename {argv[1]}; // if so, use the inputted filename
        filestream.open(filename);
    }
    
    // Configure background
    const double background_radius {10};
    Background background {background_radius, Background::image};

    // Get file
    background.load_ppm("images/vista_panorama_hres.ppm");

    // Configure viewport
    const double fov {1.815}; //1.815 rads is valorant fov, 104 degrees
    camera.set_viewport_settings(fov);

    /*
        RAY TRACING TIME BABY
    */

    // Initialize a metric,
    double black_hole_mass {1};
    Metric metric { Metric::SchwarzschildMetric, black_hole_mass };

    // Initialize paths (this sets up the paths array)
    camera.initialize_paths(metric);

    // Define the "not colliding" conditions
    std::function<bool(Path&)> within_radius = [background_radius, black_hole_mass](Path& path) -> bool {
        
        // get radius
        double radius = path.get_position()[1];

        // Collision happens when photon is outside background or close to event horizon
        bool not_collided_bool { radius < background_radius && radius > 2*black_hole_mass+0.1 };

        return not_collided_bool;
    };

    // Pathtrace until collision happens
    double dt {0.1};
    camera.pathtrace(within_radius, dt, metric);

    /*
        WRITE TO FILE
    */

    // ppm header
    filestream << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    // we iterate over each pixel and set its color accordingly
    for (int i {0}; i < image_height; ++i) {
        for (int j {0}; j < image_width; ++j) {

            // Get collision position
            Vec3 collision_pos = camera.get_paths()[i][j].get_position().get_vec3();
            Vec3 spherical_collision_pos { CoordinateSystem3::Cartesian_to_Spherical(collision_pos) };

            // std::clog << "collision is at " << spherical_collision_pos << '\n';

            // Get the ray's color and save as pixel_color
            Vec3 pixel_color = background.get_color(spherical_collision_pos);

            // Write color to output stream
            filestream << int(pixel_color[0]) << ' ' << int(pixel_color[1]) << ' ' << int(pixel_color[2]) << '\n';

        }
    }

    // Finished!
    std::clog << "\rDone.                           \n";

    filestream.close();

    return 0;
}