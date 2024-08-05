#pragma once
#include <iostream>
#include <vector>
#include "vec3.h"
#include "path.h"

// Define Camera class (including viewport stuff)
class Camera {
    private:
        // Cameras have a position 3-vector
        Vec3 position;

        // To determine camera orientation:
        // Cameras have a direction unit vector, a unit vector corresponding to "up" zhat
        Vec3 directionhat;
        Vec3 uphat; // should be orthogonal to directionhat

        // Viewport has uhat vhat unit vectors to scan across the viewport from left to right and top to bottom
        Vec3 viewport_uhat { cross(directionhat, uphat) };
        Vec3 viewport_vhat { -uphat };

        // Viewport needs image settings
        int image_width {};
        int image_height {};
        double aspect_ratio {};

        // Viewport needs a focal distance and FOV
        double viewport_field_of_view {};
        double viewport_distance {};

        // These are defined implicitly by the above variables
        double viewport_width {};
        double viewport_height {};

        // Viewport needs information to begin initializing the paths
        Vec3 viewport_origin {};
        Vec3 viewport_delta_u {};
        Vec3 viewport_delta_v {};

        // Declare the array of paths
        std::vector<std::vector<Path>> paths {};

    public:

         // Constructors
        Camera() : position {0,0,0}, directionhat {1,0,0}, uphat {0,0,1} {}
        Camera(Vec3& pos, Vec3& dir, Vec3& up) 
            : position {pos}, directionhat {unit_vector(dir)}, uphat {unit_vector(up)} { // note: autonormalizes dir and up
                try {
                    if (dot(directionhat, uphat) != 0) { //throw exception is uphat is not orthogonal to directionhat
                        throw -1;
                    }
                }
                catch (int) {
                    std::cerr << "Camera 'up' direction is not orthogonal to the camera's viewing direction.\n";
                }
        } 

        // Access functions
        std::vector<std::vector<Path>>& get_paths() { return paths; }

        // Initialize viewport stuff

        // Define image using pixel width and pixel height
        void set_image_settings(int width, int height);

        // Define viewport using fov and focal distance
        void set_viewport_settings(double fov, double dis = 1);

        // Initialize viewport
        void initialize_paths();

        // Update the uhat and vhat vectors
        void update_viewport_vectors();

        // Rotate camera
        void rotate(const double pitch_angle, const double yaw_angle, const double roll_angle);

        // Pathtrace until condition is no longer met (condition is a lambda function)
        void pathtrace(std::function<bool(Path&)> condition, const double dt);

};

Vec3 rotate_vector(const Vec3& vector, const Vec3& axis_vector, const double rotation_angle);