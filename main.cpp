/*
    THIS FILE EXISTS ONLY TO TEST OUT CERTAIN FEATURES
    BASICALLY THIS IS A DEBUGGER
*/

#include <iostream>
#include <cmath>
#include <vector>
#include "vec3.h"
#include "pathtracer.h"

void print_path(Path& path) {
    std::cout << "path has position " << path.pos << " and direction " << path.vel << ".\n";
}

void write_color(std::ostream& out, const Vec3& pixel_color) {

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

    double path_distance {path.pos.norm()};

    while (path_distance < radius) {
        path.propagate(dt);
        path_distance = path.pos.norm(); // is this inefficient?
    }

    return path.pos;
}

double g_pi = 3.14159;

Vec3 get_color(Vec3& pos) {
    double theta = atan2(sqrt(pos[0]*pos[0]+pos[1]*pos[1]),pos[2]);
    double phi = atan2(pos[1], pos[0]);
    
    // This is a fractional color, represented as a proportion
    Vec3 rgb {};

    if (std::fmod(theta,g_pi/8) > g_pi/16) {
        rgb = {1,1,1};
    }
    else {
        rgb = {1,0,0};
    }

    return rgb;
}

int main() {

    // Configure image size
    const int image_width {640};
    const int image_height {480};
    const double aspect_ratio {double(image_width)/double(image_height)};

    // Configure background
    const double background_radius {10};

    // Viewport settings
    const Vec3 camera_position {-9,0,0};
    const Vec3 camera_direction {1,0,0}; // norm of this is distance to viewport
    // These are the unit vectors for the viewport: uhat is left to right, vhat is top to bottom
    const Vec3 uhat {0, -1, 0}; // need to compute this automatically later
    const Vec3 vhat {0, 0, -1}; // need to compute this automatically later

    // Set FOV, then compute viewport size
    const double field_of_view {1.815} ; //1.815 rads is valorant fov, 104 degrees
    const double viewport_width {
        2*camera_direction.norm()*tan(field_of_view/2)
    };
    const double viewport_height {viewport_width / aspect_ratio};

    // Initialize viewport
    const Vec3 viewport_origin {
        camera_direction - (viewport_width/2)*uhat - (viewport_height/2)*vhat
    };
    const Vec3 delta_u { (viewport_width/double(image_width))*uhat };
    const Vec3 delta_v { (viewport_height/double(image_height))*vhat };

    // Initialize rays
    std::vector<std::vector<Path>> rays {};
    rays.resize(image_height);
    for (int i {0}; i < image_height; i++) {
        rays[i].resize(image_width);
    }

    for (int i {0}; i < image_height; ++i) {
        for (int j {0}; j < image_width; ++j) {

            // Initialize each ray
            rays[i][j].pos = camera_position; // preferred over Path::update_pos, since we want to overwrite the position
            rays[i][j].vel = unit_vector(viewport_origin + delta_u*0.5 + delta_v*0.5 + j*delta_u + i*delta_v);

        }
    }

    /*

        RAY TRACING TIIIIIIIME (/◕ヮ◕)/
    
    */

    // ppm header
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    // i,j are indices: i <-> rows, j <-> columns
    // we iterate over each pixel and set its color accordingly
    for (int i {0}; i < image_height; ++i) {
        
        // Progress bar
        std::clog << '\r' << int(100*(double(i)/image_height)) << "\% completed. " << std::flush;

        for (int j {0}; j < image_width; ++j) {

            // Trace a ray till it collides
            Vec3 collision_pos = get_collision_pos(rays[i][j], background_radius);

            // Get the ray's color and save as pixel_color
            Vec3 pixel_color = get_color(collision_pos);

            // Write color to output stream
            write_color(std::cout, pixel_color);

        }
    }

    // Finished!
    std::clog << "\rDone.               \n";

    return 0;
}