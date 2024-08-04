#include <iostream>
#include <cmath>
// #include <vector>
#include <fstream>
#include <string>
#include "vec3.h"
#include "camera.h"
#include "pathtracer.h"

void print_path(Path& path) {
    std::cout << "path has position " << path.position << " and direction " << path.velocity << ".\n";
}

void write_color(std::ofstream& out, const Vec3& pixel_color) {

    // pixel_color values should be in the range [0,1]
    double r_proportion {std::abs(pixel_color.e[0])};
    double g_proportion {std::abs(pixel_color.e[1])};
    double b_proportion {std::abs(pixel_color.e[2])};

    // Convert RGB proportions to the range [0,255]
    int r {int(r_proportion*255)};
    int g {int(g_proportion*255)};
    int b {int(b_proportion*255)};

    // Output RGB values to output stream
    out << r << ' ' << g << ' ' << b << '\n';
}

// Propagate path until position vector is outside of sphere
Vec3& get_collision_pos(Path& path, double radius, double dt = 0.1) {

    double path_distance {path.position.norm()};

    while (path_distance < radius) {
        path.propagate(dt);
        path_distance = path.position.norm(); // is this inefficient?
    }

    return path.position;
}

double g_pi = 3.14159;

Vec3 get_color(Vec3& pos) {
    double theta = atan2(sqrt(pos[0]*pos[0]+pos[1]*pos[1]),pos[2]);
    // double phi = atan2(pos[1], pos[0]);
    
    // This is a fractional color, represented as a proportion
    Vec3 rgb {};

    if (std::fmod(theta,g_pi/8) > g_pi/16) {
        rgb = {1,0,0};
    }
    else {
        rgb = {1, 1, 1};
    }

    return rgb;
}

int main(int argc, char *argv[]) {

    // Configure camera position and direction
    Vec3 camera_position {-9,0,0};
    Vec3 camera_direction {1,0,0};
    Vec3 camera_up {0,0,1};
    Camera camera {camera_position, camera_direction, camera_up};

    // Rotate camera!
    camera.rotate(g_pi/4,g_pi/4,g_pi/4);

    // Configure image size
    const int image_width {640};
    const int image_height {480};
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

    // Configure viewport
    const double fov {1.815}; //1.815 rads is valorant fov, 104 degrees
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

            // Get the ray's color and save as pixel_color
            Vec3 pixel_color = get_color(collision_pos);

            // Write color to output stream
            write_color(filestream, pixel_color);

        }
    }

    // Finished!
    std::clog << "\rDone.               \n";

    filestream.close();

    return 0;
}