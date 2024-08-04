#pragma once
#include <vector>
#include <string>
#include <cmath>
#include "vec3.h"

class Background {
    public:

        // We want the "type" enum to be public
        enum Type {
            layered,    // This is the red/white layered sphere
            image       // This is if the background is an image
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

    public:

        // Constructors
        Background() : radius {10}, type {layered} {}
        Background(const double radius, Type type) : radius {radius}, type {type} {}

        // Access functions
        std::vector<std::vector<Vec3>>& get_image_array() { return image_array; }

        // Load image into memory
        void load_ppm(const std::string filename);

        // Get color given a spherical coordinate
        Vec3& get_color(Vec3& spherical_coordinates);

        // define pi
        double m_pi { 3.14159 };

};