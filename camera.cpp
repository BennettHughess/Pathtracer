#include <cmath>
#include "camera.h"

// Set image variables in the Camera class
void Camera::set_image_settings(int width, int height) {
    image_width = width;
    image_height = height;
    aspect_ratio = double(image_width)/double(image_height);
}

// Set viewport variables in the Camera class
void Camera::set_viewport_settings(double fov, double dis) {
    viewport_field_of_view = fov;
    viewport_distance = dis;

    viewport_width = 2*viewport_distance*tan(viewport_field_of_view/2);
    viewport_height = viewport_width / aspect_ratio;
}

// Initialize rays
void Camera::initialize_rays() {

    // Initialize vectors to begin iterating over viewport
    viewport_origin = directionhat*viewport_distance - (viewport_width/2)*viewport_uhat - (viewport_height/2)*viewport_vhat;
    viewport_delta_u = (viewport_width/double(image_width))*viewport_uhat;
    viewport_delta_v = (viewport_height/double(image_height))*viewport_vhat;

    // Resize rays array
    rays.resize(image_height);
    for (int i {0}; i < image_height; ++i) {
        rays[i].resize(image_width);
    }

    // Initialize the rays array with each ray pointing out of the camera and through the viewport
    for (int i {0}; i < image_height; ++i) {
        for (int j {0}; j < image_width; ++j) {

            // Initialize each ray
            rays[i][j].position = position; // preferred over Path::update_position, since we want to overwrite the position
            rays[i][j].velocity = unit_vector(viewport_origin + viewport_delta_u*0.5 
            + viewport_delta_v*0.5 + j*viewport_delta_u + i*viewport_delta_v);

        }
    }

}

void Camera::update_viewport_vectors() {
    viewport_uhat = cross(directionhat, uphat); //note that uhat is used to build a basis for R^3
    viewport_vhat = -uphat;
}

// Rotate camera
void Camera::rotate(double pitch_angle, double yaw_angle, double roll_angle) {
    // First we pitch the camera, then yaw, then roll. Order matters!
    /* Rotation direction corresponds to right-hand-rule i.e. positive 
    pitch angle rotates the viewing direction clockwise about the uhat vector */

    // First pitch our basis vectors (rotate about uhat)
    directionhat = rotate_vector(directionhat, viewport_uhat, pitch_angle);
    uphat = rotate_vector(uphat, viewport_uhat, pitch_angle);

    // Now yaw our basis vectors (rotate about uphat)
    directionhat = rotate_vector(directionhat, uphat, yaw_angle);
    viewport_uhat = rotate_vector(viewport_uhat, uphat, yaw_angle); //only need to update uhat here

    // Now roll our basis vectors (rotate about directionhat)
    uphat = rotate_vector(uphat, directionhat, roll_angle);
    update_viewport_vectors(); // need to update uhat and vhat
}

Vec3 rotate_vector(const Vec3& vector, const Vec3& axis_vector, const double rotation_angle) {
    // We use Rodrigues' rotation formula:
    // https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    Vec3 rotated_vector {
        vector*cos(rotation_angle) + cross(axis_vector,vector)*sin(rotation_angle) 
        + axis_vector * dot(axis_vector,vector) * (1 - cos(rotation_angle))
    };
    return rotated_vector;
}