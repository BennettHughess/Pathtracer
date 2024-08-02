#include <iostream>
#include <cmath>
#include "vec3.h"

/*
    FUNCTIONS
*/

// Write pixel color to output stream
// note: this is configured to write to a .ppm file
void write_color(std::ostream& out, const Vec3& pixel_color) {

    // pixel_color values should be in the range [0,1]
    double r_proportion {pixel_color.e[0]};
    double g_proportion {pixel_color.e[1]};
    double b_proportion {pixel_color.e[2]};

    // Convert RGB proportions to the range [0,255]
    int r {int(r_proportion*255)};
    int g {int(g_proportion*255)};
    int b {int(b_proportion*255)};

    // Output RGB values to output stream
    out << r << ' ' << g << ' ' << b << '\n';
}

/*
    MAIN
*/

int main(){

    // Configure image size
    const int image_width {1920};
    const int image_height {1080};

    /*
        WRITE COLOR TO FILE
        note: file type is supposed to be .ppm
    */
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    // i,j are indices: i <-> rows, j <-> columns
    // we iterate over each pixel and set its color accordingly
    for (int i {0}; i < image_height; ++i) {
        
        // Progress bar
        std::clog << '\r' << int(100*(double(i)/image_height)) << "\% completed. " << std::flush;

        for (int j {0}; j < image_width; ++j) {

            // Calculate pixel color
            Vec3 pixel_color {
                double(i)/(image_height),
                double(j)/(image_width),
                127
            };

            // Write color to output stream
            write_color(std::cout, pixel_color);

        }
    }

    // Finished!
    std::clog << "\rDone.               \n";
    return 0;
}