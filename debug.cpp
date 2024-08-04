#include "background.h"
#include <string>
#include <vector>
#include "vec3.h"
#include <iostream>

int main() {

    // Get file
    std::string filename {"images/smileyface.ppm"};

    std::cout << "filename set \n";

    // Declare background
    Background background {10, Background::image};

    std::cout << "background initialized. doing something before we load ppm: \n";

    // Load the file
    background.load_ppm(filename);

    std::cout << "ppm loaded \n";

    // Grab image array
    std::vector<std::vector<Vec3>>& image = background.get_image_array();

    std::cout << "image array grabbed \n";

    // Print some of image array into terminal
    for (int i {0}; i < 100; ++i) {
        for (int j {0}; j < 100; ++j) {
            std::cout << image[i][j];
        }
    }

    return 0;
}