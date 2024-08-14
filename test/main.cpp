#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <functional>
#include "../include/vec3.h"
#include "../include/vec4.h"
#include "../include/camera.h"
#include "../include/path.h"
#include "../include/background.h"

int main(int argc, char *argv[]) {

    // Camera position and direction are in cartesian (x,y,z) coordinates
    Vec3 camera_position {0,-20,0};
    Vec3 camera_direction {0,1,0};
    Vec3 camera_up {0,0,1};
    Camera camera {camera_position, camera_direction, camera_up};

    //double pi = 3.141459;
    // Rotate camera!
    //camera.rotate(0,-0.3,0);

    // Configure image size
    const int image_width {100};
    const int image_height {100};
    camera.set_image_settings(image_width, image_height);

    // Get filename and initialize filestream
    std::ofstream filestream;
    if (argv[1] == NULL) {        // check if filename was inputted
        filestream.open("../main.ppm"); // if not, default output file to main.ppm
    } 
    else {
        std::string filename {argv[1]}; // if so, use the inputted filename
        filestream.open(filename);
    }
    
    // Configure background
    const double background_radius {30};
    Background background {background_radius, Background::image};

    // Get file
    background.load_ppm("../images/milky_way_hres.ppm");

    // Configure viewport
    const double fov {1.815}; //1.815 rads is valorant fov, 104 degrees
    camera.set_viewport_settings(fov);

    /*
        RAY TRACING TIME BABY
    */

    // Initialize a metric,
    double black_hole_mass {1};
    Metric metric { Metric::SchwarzschildMetric, black_hole_mass };

    // Configure the integrator (with tolerances as necessary)
    Path::Integrator integrator {Path::CashKarp};       // integrator is adaptive and uses *fractional* error!
    double dlam {0.01};
    double max_dlam {1};
    double min_dlam {1.0E-20};                        // basically 0 min_dlam
    double tolerance {0.000000000001};             // small tolerance, approx 10^-15

    // Initialize paths (this sets up the paths array)
    camera.initialize_paths(metric, integrator, max_dlam, min_dlam, tolerance);

    // Define the "not colliding" conditions. we pass this to the pathtracer to know when to stop pathtracing.
    std::function<bool(Path&)> collision_checker = [background_radius, black_hole_mass](Path& path) -> bool {
        
        // get radius
        double radius = path.get_position()[1];

        // Collision happens when photon is outside background
        bool inside_background { radius < background_radius };

        // or close to event horizon
        bool far_from_event_horizon { radius > 2.01*black_hole_mass} ;

        return inside_background && far_from_event_horizon;
    };

    // Pathtrace until a collision happens
    camera.pathtrace(collision_checker, dlam, metric);

    std::cout << "writing to file!" << '\n';

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
            //Vec3 spherical_collision_pos = CoordinateSystem3::Cartesian_to_Spherical(collision_pos);

            // std::clog << "collision for " << i << ' ' << j << " is at " << collision_pos << '\n';

            // Get the ray's color and save as pixel_color
            Vec3 pixel_color = background.get_color(collision_pos); // NOTE what is passed depends on the metric of choice

            // Write color to output stream
            filestream << int(pixel_color[0]) << ' ' << int(pixel_color[1]) << ' ' << int(pixel_color[2]) << '\n';

        }
    }

    // Finished!
    std::clog << "\rDone.                           \n";

    filestream.close();

    return 0;
}