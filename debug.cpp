#include "background.h"
#include <string>
#include <vector>
#include "vec3.h"
#include <iostream>

int main() {

    // Get file
    std::string filename {"images/vista_panorama.ppm"};

    std::cout << "filename set \n";

    // Declare background
    Background background {10, Background::image};

    std::cout << "background initialized. doing something before we load ppm: \n";

    // Load the file
    background.load_ppm(filename);

    std::cout << "ppm loaded \n";

    // Print the file
    background.save_ppm("debug.ppm"); //prints fine!

    return 0;
}