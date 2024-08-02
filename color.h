#pragma once
#include <iostream>
#include "vec3.h"

// Write pixel color to output stream
// note: this is configured to write to a .ppm fiel
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
