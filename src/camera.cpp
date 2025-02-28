#include <cmath>
#include <iostream>
#include "../include/camera.h"
#include "../include/metric.h"
#include "omp.h"        // This is from OpenMP, need to pass -fopenmp as a compiler flag to use it

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

// Initialize paths
void Camera::initialize_paths(Metric& metric, Path::Integrator integrator, double max_dlam, double min_dlam, double tolerance) {

    // Initialize vectors to begin iterating over viewport
    viewport_origin = directionhat*viewport_distance - (viewport_width/2)*viewport_uhat - (viewport_height/2)*viewport_vhat;
    viewport_delta_u = (viewport_width/double(image_width))*viewport_uhat;
    viewport_delta_v = (viewport_height/double(image_height))*viewport_vhat;

    // Resize paths array
    paths.resize(image_height);
    for (int i {0}; i < image_height; ++i) {
        paths[i].resize(image_width);
    }

    // Get metric type for the coming switch statement
    Metric::MetricType metrictype { metric.get_type() };

    // Declare initial position and initial velocity direction
    Vec3 initial_position {};
    Vec3 adapted_unit_direction{};

    // Figure out which coordinate system to use for which metric
    switch (metrictype) {

        case Metric::CartesianMinkowskiMetric:
            initial_position = position;
            break;

        case Metric::SphericalMinkowskiMetric:
            initial_position = CoordinateSystem3::Cartesian_to_Spherical(position);
            break;

        case Metric::SchwarzschildMetric:
            initial_position = CoordinateSystem3::Cartesian_to_Spherical(position);
            break;

        case Metric::CartesianIsotropicSchwarzschildMetric:
            initial_position = position;
            break;

    }

    // Initialize the paths array with each ray pointing out of the camera and through the viewport
    for (int i {0}; i < image_height; ++i) {
        for (int j {0}; j < image_width; ++j) {

            // Path will begin at t=0 at the already determined initial position (camera's coordinates in different systems)
            Vec4 path_position { Vec4{0, initial_position} };

            // std::cout << "camera.cpp: paths are starting at " << path_position << '\n';

            // initial position is in spherical coordinates
            paths[i][j].set_position( path_position ); 

            // paths will point in the direction of the pixel
            Vec3 unit_direction { unit_vector(viewport_origin + viewport_delta_u*0.5 
                + viewport_delta_v*0.5 + j*viewport_delta_u + i*viewport_delta_v) };

            // convert from camera coordinates to spacetime coordinates
            switch (metrictype) {

                case Metric::CartesianMinkowskiMetric: 
                    adapted_unit_direction = unit_direction;
                    break;

                case Metric::SphericalMinkowskiMetric:
                    adapted_unit_direction = CoordinateSystem3::CartesianTangent_to_SphericalTangent(position, unit_direction);
                    break;

                case Metric::SchwarzschildMetric:
                    adapted_unit_direction = CoordinateSystem3::CartesianTangent_to_SphericalTangent(position, unit_direction);
                    break;

                case Metric::CartesianIsotropicSchwarzschildMetric: 
                    adapted_unit_direction = unit_direction;
                    break;

            }

            // path with null velocity (speed of light)
            paths[i][j].set_velocity( convert_to_null(adapted_unit_direction, path_position, metric) );

            // set path with an integrator
            paths[i][j].set_integrator(integrator);
            paths[i][j].set_max_dlam(max_dlam);
            paths[i][j].set_min_dlam(min_dlam);
            paths[i][j].set_tolerance(tolerance);

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

// Iterate through the paths array and pathtrace each one until condition is no longer met
void Camera::pathtrace(std::function<bool(Path&)> condition, const double dlam, Metric& metric) {

    std::clog << "Beginning pathtrace! \n";

    if (multithreaded) {

        omp_set_num_threads(threads);
        // Loop through image
        for (int i {0}; i < image_height; ++i) {

            // Progress bar
            std::clog << "\rPathtrace is " << int(100*(double(i)/image_height)) << "% completed. " 
                << "Working on row " << i << " of " << image_height << "." << std::flush;

            // Run in parallel
            #pragma omp parallel for
            for (int j = 0; j < image_width; ++j) {

                // Trace a ray till it collides
                paths[i][j].loop_propagate(condition, dlam, metric);

            }
        }
    }
    else {
        // Loop through image
        for (int i {0}; i < image_height; ++i) {

            // Progress bar
            std::clog << "\rPathtrace is " << int(100*(double(i)/image_height)) << "% completed. " 
                << "Working on row " << i << " of " << image_height << "." << std::flush;

            // Run in parallel
            for (int j = 0; j < image_width; ++j) {

            // Trace a ray till it collides
            paths[i][j].loop_propagate(condition, dlam, metric);

            }
        }
    }

    std::clog << '\n';
}