#include <fstream>
#include <iostream>
#include "background.h"

void Background::load_ppm(const std::string filename) {

    if (type == image) {

        // Open filestream
        std::ifstream filestream;
        filestream.open(filename);

        // Doesn't throw an exception, just points it out to you.
        if (!filestream) {
            std::cerr << "Could not open file. \n";
        }

        // Skip first line (should be "P3")
        std::string throwaway_line {};
        std::getline(filestream, throwaway_line);

        // Read the width and height (next two words)
        filestream >> image_width;
        filestream >> image_height;

        // Resize image_array with the image_height and image_width
        image_array.resize(image_height);
        for (int i {0}; i < image_height; ++i) {
            image_array[i].resize(image_width);
        }

        // Skip next line (is just 255)
        filestream >> throwaway_line;

        // Declare some variables for the coming loop
        std::string color {};
        int r {};
        int g {};
        int b {};
        Vec3 rgb {};

        // Read through rest of document in pairs of threes and save to image_array:
        for (int i {0}; i < image_height; ++i) {
            for (int j {0}; j < image_width; ++j) {

                // Take r, g, b as it comes out of the stream
                filestream >> color;
                r = std::stoi(color);

                filestream >> color;
                g = std::stoi(color);

                filestream >> color;
                b = std::stoi(color);

                // Save r, g ,b to a vector
                Vec3 rgb {double(r), double(g), double(b)};

                // Save vector into the image array
                image_array[i][j] = rgb;

            }
        }

        // Close filestream
        filestream.close();

        // To make sure that we know we have an image to load
        image_been_loaded = true;

    }
    else {
        std::cerr << "Background is not of type 'image'.";
    }
}

// Save image_array to a ppm file
void Background::save_ppm(const std::string filename) {

    if (image_been_loaded && type == image) {

        // Open filestream
        std::ofstream filestream;
        filestream.open(filename);

        // Doesn't throw an exception, just points it out to you.
        if (!filestream) {
            std::cerr << "Could not open file. \n";
        }

        // Print header
        filestream << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        // Output image_array to filestream
        for (int i {0}; i < image_height; ++i) {
            for (int j {0}; j < image_width; ++j) {

                // Write color to output stream
                filestream << int(image_array[i][j][0]) << ' ' << int(image_array[i][j][1]) << ' ' << int(image_array[i][j][2]) << '\n';

            }
        }

    }
    else {
         std::cerr << "Image was not loaded prior to saving. \n";
    }

}

Vec3 Background::get_color(Vec3& spherical_coordinates) {
    // note: spherical coordinates should be in r, theta, phi

    if (image_been_loaded && type == image) {

        // We project the image onto the sphere using equirectangular projection
        // The size of each pixel is

        int x_pixel = int(spherical_coordinates[2]*image_width/(2*m_pi));
        int y_pixel = int(spherical_coordinates[1]*image_height/m_pi);

        Vec3 color { image_array[y_pixel][x_pixel] };

        // std::cout << "color got: " << color << '\n';

        return color;

    }
    else if (type == layered) {
        
        if (fmod(spherical_coordinates[1],m_pi/8)> m_pi/16) {
            Vec3 color { 255, 0, 0 };
            return color;
        }
        else if (fmod(spherical_coordinates[2],m_pi/8)> m_pi/16) {
            Vec3 color { 0, 255, 0 };
            return color;
        }
        else {
            Vec3 color { 255, 255, 255 };
            return color;
        }

    }
    else {

        std::cerr << "Image was not loaded prior to raytracing or unknown type. \n";
        Vec3 color { 0, 0, 0 };
        return color;

    }
}