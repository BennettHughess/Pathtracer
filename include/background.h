#pragma once
#include <vector>
#include <string>
#include <cmath>
#include "vec3.h"

class Background {
    public:

        // We want the "type" enum to be public
        enum Type {
            Layered,    // This is the red/white layered sphere
            Image       // This is if the background is an image
        };

    private:
        // Background sphere will always have a radius
        const double radius;

        // A background sphere can have different "looks"
        Type type;

        // If the background is an image, it needs to store the image in memory as Vec3s:
        std::vector<std::vector<Vec3>> image_array {};

        // To store if an image has been loaded
        bool image_been_loaded {false};

        // If the background is an image, it also needs to know the image dimensions:
        int image_height {};
        int image_width {};
        int image_channels {};        // used for the std_image lib

    public:

        // Constructors
        Background() : radius {10}, type {Layered} {}
        Background(const double r, Type t) : radius {r}, type {t} {}

        // Access functions
        std::vector<std::vector<Vec3>>& get_image_array() { return image_array; }
        void set_image_array(const std::vector<std::vector<Vec3>> array);
        void set_pixel_color(Vec3 rgb, int i, int j) { image_array[i][j] = rgb;  }

        // Load ppm into memory
        void load_ppm(const std::string filename);

        // Load any supported kind of image (png, jpg, etc.)
        void load_img(const std::string filename);

        // Save loaded ppm, png with data from arrayinto a file
        void save_ppm(const std::string filename);
        void save_png(const std::string filename, const std::vector<std::vector<Vec3>> array);

        // Get color given a spherical coordinate
        Vec3 get_color(Vec3& spherical_coordinates);

        // define pi
        double m_pi { 3.14159265359 };

};