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
        int max_color_value {};
        std::string max_color_value_string {};
        filestream >> max_color_value_string;
        max_color_value = std::stoi(max_color_value_string);

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
                r = int(std::stoi(color)*255/max_color_value);          // compress to a 255 maximum color value

                filestream >> color;
                g = int(std::stoi(color)*255/max_color_value);

                filestream >> color;
                b = int(std::stoi(color)*255/max_color_value);

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

    // normalize coordinates
    Vec3 normalized_spherical_coordinates { CoordinateSystem3::Normalize_SphericalVector(spherical_coordinates) };

    // get r, theta, phi
    double r { normalized_spherical_coordinates[0] };
    double theta { normalized_spherical_coordinates[1] };
    double phi { normalized_spherical_coordinates[2] };

    // std::cout << "background.cpp: get_color recieving coordinates " << r << ' ' << theta << ' ' << phi << '\n';

    Vec3 color;

    if (image_been_loaded && type == image) {

        // Check if collision was at background
        if (r >= radius
            && theta >= 0
            && theta < m_pi
            && phi >= 0
            && phi < 2*m_pi) {

            // We project the image onto the sphere using equirectangular projection
            int x_pixel = int(phi*image_width/(2*m_pi));
            int y_pixel = int(theta*image_height/m_pi);

            color = { image_array[y_pixel][x_pixel] };
                
        }
        // if collision is weird, use error color (green)
        else if (r < 0 or theta > m_pi or theta < m_pi or phi < 0 or phi > 2*m_pi) {
            color = {0, 255, 0};
        }
        // if not weird but less than radius (for example, if at event horizon) then set to black
        else {
            color = {0,0,0};
        }
        
    }
    else if (type == layered) {
        
        if (fmod(theta, m_pi/8) > m_pi/16) {
            color = { 255, 0, 0 };
        }
        else if (fmod(phi, m_pi/8)> m_pi/16) {
            color = { 0, 255, 0 };
        }
        else {
            color = { 255, 255, 255 };
        }

    }
    else {

        std::cerr << "Image was not loaded prior to raytracing or unknown type. \n";
        color = { 0, 0, 0 };

    }
    return color;
}