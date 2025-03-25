#include <fstream>
#include <iostream>
#include "../include/background.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image.h"
#include "../lib/stb_image_write.h"

void Background::load_ppm(const std::string filename) {

    if (type == Image) {

        // Open filestream
        std::ifstream filestream;
        try {
            filestream.open(filename);
            if (!filestream.is_open()) {
                throw 12;
            }
        }   
        catch (int Err) {
            std::cerr << "ERROR " << Err << ": Background image file failed to be opened. Check path?" << std::endl;
            throw Err;
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
                rgb = {double(r), double(g), double(b)};

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

void Background::load_img(const std::string filename) {

    if (type == Image) {

        // Check if png can be opened.
        try {
            if (!stbi_info(filename.c_str(), &image_width, &image_height, &image_channels)) {
                throw 12;
            }
        }
        catch (int Err) {
            std::cerr << "ERROR " << Err << ": Background image file failed to be opened. Check path?" << std::endl;
            throw Err;
        }

        // Save png to a file
        unsigned char* data = stbi_load(filename.c_str(), &image_width, &image_height, &image_channels, 0);

        // Resize image_array with the image_height and image_width
        image_array.resize(image_height);
        for (int i {0}; i < image_height; ++i) {
            image_array[i].resize(image_width);
        }

        // Create rgb vector for later
        Vec3 rgb;

        // Read RGB values and save to image_array
        for (int i {0}; i < image_height; ++i) {
            for (int j {0}; j < image_width; ++j) {

                // Calculate pixel index
                int index = (i * image_width + j) * image_channels;

                // Get RGB values
                unsigned char r = data[index];
                unsigned char g = data[index + 1];
                unsigned char b = data[index + 2];
        
                // Save r, g ,b to a vector
                rgb = {(double)r, (double)g, (double)b};

                // Save vector into the image array
                image_array[i][j] = rgb;

            }
        }

        // Free image data
        stbi_image_free(data);

        // To make sure that we know we have an image to load
        image_been_loaded = true;

    }
    else {
        std::cerr << "Background is not of type 'image'.";
        throw 1;
    }
}

// Save image_array to a ppm file
void Background::save_ppm(const std::string filename) {

    if (image_been_loaded && type == Image) {

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

// Save array to a png file
void Background::save_png(const std::string filename, const std::vector<std::vector<Vec3>> array) {

    int array_height = array.size();
    int array_width = array[0].size();
    int channels = 3;   // rgb channels

    // Allocate memory for the image
    unsigned char* data = new unsigned char[array_width * array_height * channels];

    // define some vars for the loop
    int index, r, g, b;
    unsigned char char_r, char_g, char_b;

    // Set data struct equal to array
    for (int i = 0; i < array_height; i++) {
        for (int j = 0; j < array_width; j++) {

            // get index of pixel
            index = (i * array_width + j) * channels;

            // get color at current pixel
            r = array[i][j][0];
            g = array[i][j][1];
            b = array[i][j][2];

            // convert r,g,b to char for saving
            char_r = static_cast<unsigned char>(r);
            char_g = static_cast<unsigned char>(g);
            char_b = static_cast<unsigned char>(b);

            // set data to pixel color
            data[index] = char_r;
            data[index + 1] = char_g;
            data[index + 2] = char_b;
        }
    }

    // Save as png
    try {
        if (!stbi_write_png(filename.c_str(), array_width, array_height, channels, data, array_width * channels)) {
            delete[] data;
            throw 40;
        }
    }
    catch (int Err) {
        std::cerr << "ERROR " << Err << ": Failed to save file as png." << std::endl;
        throw Err;
    }

    // Free memory
    delete[] data;

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

    if (image_been_loaded && type == Image) {

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
        else if (r < 0 or theta > m_pi or theta < 0 or phi < 0 or phi > 2*m_pi) {
            std::cout << "background.cpp: weird collision at " << r << ' ' << theta << ' ' << phi << '\n';
            color = {0, 255, 0};
        }
        // if not weird but less than radius (for example, if at event horizon) then set to black
        else {
            color = {0,0,0};
        }
        
    }
    else if (type == Layered) {
        
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

void Background::set_image_array(const std::vector<std::vector<Vec3>> array) {

    int array_width = array.size();
    int array_height = array[0].size();

    try {
        if (array_width != image_width or array_height != image_height) {
            throw 29;
        }
    }
    catch (int Err) {
        std::cerr << "ERROR " << Err << ": Cant set image array equal to array; mismatched dimensions. " 
            << "image_width=" << image_width << " image_height=" << image_height 
            << " array_width=" << array_width << " array_height=" << array_height << std::endl;
        throw Err;
    }

    for (int i {0}; i < image_height; ++i) {
        for (int j {0}; j < image_width; ++j) {

            image_array[i][j] = array[i][j];

        }
    }
}