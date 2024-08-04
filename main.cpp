#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include "vec3.h"
#include "camera.h"
#include "pathtracer.h"
#include "background.h"

// Propagate path until position vector is outside of sphere
Vec3& get_collision_pos(Path& path, double radius, double dt = 0.1) {

    double path_distance {path.position.norm()};

    while (path_distance < radius) {
        path.propagate(dt);
        path_distance = path.position.norm(); // is this inefficient?
    }

    return path.position;
}

const double g_pi = 3.14159;

Vec3 cartesian_to_spherical(const Vec3& cartesian) {

    double theta { atan2(sqrt(cartesian[0]*cartesian[0]+cartesian[1]*cartesian[1]),cartesian[2]) };

    double phi {};
    if (cartesian[0] >= 0 && cartesian[1] >= 0) {
        phi = atan(cartesian[1]/cartesian[0]);
    }
    else if (cartesian[0] < 0 && cartesian[1] >= 0) {
        phi = g_pi/2 + atan(-cartesian[0]/cartesian[1]);
    }
    else if (cartesian[0] < 0 && cartesian[1] < 0) {
        phi = g_pi + atan(cartesian[1]/cartesian[0]);
    }
    else if (cartesian[0] >= 0 && cartesian[1] < 0) {
        phi = 3*g_pi/2 + atan(-cartesian[0]/cartesian[1]);
    }
    else {
        phi = 0;
    }

    double r { cartesian.norm() };

    return Vec3 {r, theta, phi};
}

int main(int argc, char *argv[]) {

    // Configure camera position and direction
    Vec3 camera_position {0,0,-5};
    Vec3 camera_direction {-1,0,0};
    Vec3 camera_up {0,0,1};
    Camera camera {camera_position, camera_direction, camera_up};

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
    background.load_ppm("images/vista_panorama.ppm");

    // Configure viewport
    const double fov {2.0}; //1.815 rads is valorant fov, 104 degrees
    camera.set_viewport_settings(fov);

    // Initialize rays (this sets up the rays array)
    camera.initialize_rays();

    /*
        RAY TRACING TIIIIIIIME (/◕ヮ◕)/
    */

    // ppm header
    filestream << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    // i,j are indices: i <-> rows, j <-> columns
    // we iterate over each pixel and set its color accordingly
    for (int i {0}; i < image_height; ++i) {

        // Progress bar
        std::clog << '\r' << int(100*(double(i)/image_height)) << "\% completed. " << std::flush;

        for (int j {0}; j < image_width; ++j) {

            // Trace a ray till it collides
            Vec3 collision_pos = get_collision_pos(camera.get_rays()[i][j], background_radius);
            Vec3 spherical_collision_pos { cartesian_to_spherical(collision_pos) };

            // Get the ray's color and save as pixel_color
            Vec3 pixel_color = background.get_color(spherical_collision_pos);

            // Write color to output stream
            filestream << int(pixel_color[0]) << ' ' << int(pixel_color[1]) << ' ' << int(pixel_color[2]) << '\n';
            // filestream << spherical_collision_pos << '\n';

        }
    }

    // Finished!
    std::clog << "\rDone.               \n";

    filestream.close();

    return 0;
}