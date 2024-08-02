#include <iostream>
#include <cmath>
#include "vec3.h"
#include "color.h"

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
                std::abs(std::sin((double(i+j)/75)))
            };

            // Write color to output stream
            write_color(std::cout, pixel_color);

        }
    }

    // Finished!
    std::clog << "\rDone.               \n";
    return 0;
}