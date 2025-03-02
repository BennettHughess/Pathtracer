#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <functional>
#include <filesystem>
#include "../include/vec3.h"
#include "../include/vec4.h"
#include "../include/camera.h"
#include "../include/path.h"
#include "../include/background.h"
#include "../lib/json.hpp"

// for convenience
using json = nlohmann::json;

int main(int argc, char *argv[]) {

    // get directory location
    std::filesystem::path executable_path = std::filesystem::current_path();
    std::filesystem::path dir_path = executable_path/"..";

    // get config and image path
    std::filesystem::path config_path = dir_path/"config.json";
    std::filesystem::path image_path = dir_path/"main.ppm";

    /*
        Parse config file
    */

    // attempt to open config file
    std::ifstream configstream;
    json config;
    try {
        configstream.open(config_path.string());
        if (!configstream.is_open()) {
            throw 10;
        }
    }
    catch(int Err) {
        std::cerr << "ERROR " << Err << ": Config stream failed to open. Check path?"<< std::endl;
        std::cerr << "  Attempted to open file: " << config_path.string() << std::endl;
        return 1;
    }

    // attempt to parse config file
    try {
        config = json::parse(configstream);
    }
    catch(int Err) {
        std::cerr << "ERROR " << Err << ": Config failed to be parsed as json. Check path?" << std::endl;
        return 1;
    }
    
    // Camera position and direction are in cartesian (x,y,z) coordinates.
    // read camera position and direction from config
    Vec3 camera_position {config["camera"]["position"].template get<std::vector<double>>()};
    Vec3 camera_direction {config["camera"]["direction"].template get<std::vector<double>>()};
    Vec3 camera_up {config["camera"]["up"].template get<std::vector<double>>()};
    Camera camera {camera_position, camera_direction, camera_up};

    // Rotate camera!
    double rotate_pitch {config["camera"]["rotation"][0]};
    double rotate_yaw {config["camera"]["rotation"][1]};
    double rotate_roll {config["camera"]["rotation"][2]};
    camera.rotate(rotate_pitch,rotate_yaw,rotate_roll);

    // Configure image size
    const int image_width {config["camera"]["image"]["width"]};
    const int image_height {config["camera"]["image"]["height"]};
    camera.set_image_settings(image_width, image_height);

    // Get filename and initialize filestream
    std::ofstream filestream;
    try {
        if (argc == 1) {        // check if filename was inputted
            filestream.open(image_path.string()); // if not, default output file to main.ppm
        } 
        else {
            std::string filename {argv[1]}; // if so, use the inputted filename
            filestream.open(filename);
        }
        if (!filestream.is_open()) {
            throw 11;
        }
    }
    catch (int Err) {
        std::cerr << "ERROR " << Err << ": Image file failed to be opened. Check path?" << std::endl;
        return 1;
    }
    
    // Configure background
    const double background_radius {config["background"]["radius"]};
    std::string str_background_type { config["background"]["type"] };

    Background::Type background_type {};
    if (str_background_type == "Image") {
        background_type = Background::Type::Image;
    }
    else if (str_background_type == "Layered") {
        background_type = Background::Type::Layered;
    }
    else {
        std::cerr << "Config file: unknown background type." <<'\n';
        background_type = Background::Type::Layered;
    }

    Background background {background_radius, background_type};

    // Load background image
    try {
        background.load_ppm(config["background"]["image_path"]);
    }
    catch (int Err) {
        std::cerr << "ERROR " << Err << ": Failure in background.load_ppm. " 
            << "Attempted to open " << config["background"]["image_path"] << std::endl;
        return 1;
    }
    
    // Configure viewport
    const double viewport_fov {config["camera"]["viewport"]["fov"]}; //1.815 rads is valorant fov, 104 degrees
    const double viewport_distance {config["camera"]["viewport"]["distance"]};
    camera.set_viewport_settings(viewport_fov, viewport_distance);

    // Initialize a metric,
    double black_hole_mass {config["metric"]["black_hole_mass"]};
    std::string str_metric_type {config["metric"]["type"]};

    Metric::MetricType metric_type {};
    if (str_metric_type == "CartesianMinkowskiMetric") {
        metric_type = Metric::MetricType::CartesianMinkowskiMetric;
    }
    else if (str_metric_type == "SphericalMinkowskiMetric") {
        metric_type = Metric::MetricType::SphericalMinkowskiMetric;
    }
    else if (str_metric_type == "SchwarzschildMetric") {
        metric_type = Metric::MetricType::SchwarzschildMetric;
    }
    else if (str_metric_type == "CartesianIsotropicSchwarzschildMetric") {
        metric_type = Metric::MetricType::CartesianIsotropicSchwarzschildMetric;
    }
    else {
        std::cerr << "Config file: unknown metric type." <<'\n';
        metric_type = Metric::MetricType::CartesianMinkowskiMetric;
    }

    Metric metric { metric_type, black_hole_mass };

    // Configure the integrator (with tolerances as necessary)
    std::string str_integrator_type {config["integrator"]["type"]};

    Path::Integrator integrator_type {};
    if (str_integrator_type == "Euler") {
        integrator_type = Path::Integrator::Euler;
    }
    else if (str_integrator_type == "Verlet") {
        integrator_type = Path::Integrator::Verlet;
    }
    else if (str_integrator_type == "RK4") {
        integrator_type = Path::Integrator::RK4;
    }
    else if (str_integrator_type == "RKF45") {
        integrator_type = Path::Integrator::RKF45;
    }
    else if (str_integrator_type == "CashKarp") {
        integrator_type = Path::Integrator::CashKarp;
    }
    else {
        std::cerr << "Config file: unknown integrator type." <<'\n';
        integrator_type = Path::Integrator::Euler;
    }
    
    Path::Integrator integrator {integrator_type};    
    double dlam {config["integrator"]["dlam"]};
    double max_dlam {config["integrator"]["max_dlam"]};
    double min_dlam {config["integrator"]["min_dlam"]}; 
    double tolerance {config["integrator"]["tolerance"]};  

    // set parallel computing stuff
    /*
        Parallel_type takes on values 0, 1, 2 
            0: single thread, processed on cpu 
            1: multi thread, processed on cpu 
            2: processed on gpu
    */
    int parallel_type {config["parallel"]["parallel_type"]};
    int threads {config["parallel"]["threads"]};
    camera.set_parallel_type(parallel_type);
    camera.set_threadnum(threads);

    /*
        PATH TRACING TIMEEEEEE
    */

    // Initialize paths (this sets up the paths array)
    camera.initialize_paths(metric, integrator, max_dlam, min_dlam, tolerance);

    // Define the "not colliding" conditions. we pass this to the pathtracer to know when to stop pathtracing.
    std::function<bool(Path&)> collision_checker = [background_radius, black_hole_mass](Path& path) -> bool {
        
        /*  // CODE FOR SCHWARZSCHILD CHECKER
        // get radius
        double radius = path.get_position()[1];

        // Collision happens when photon is outside background
        bool inside_background { radius < background_radius };

        // or close to event horizon
        bool far_from_event_horizon { radius > 2.01*black_hole_mass} ;

        return inside_background && far_from_event_horizon;
        */

        // CODE FOR ISOTROPIC CARTESIAN SCHWARZSCHILD
        Vec4 pos = path.get_position();
        double rho = sqrt(pos[1]*pos[1] + pos[2]*pos[2] + pos[3]*pos[3]);
        double rho_s = 2.*black_hole_mass/4;
        double radius = rho*pow((1 + rho_s/rho),2);

        // Collision happens when photon is outside background
        bool inside_background { radius < background_radius };

        // or close to event horizon
        bool far_from_event_horizon { rho > 1.01*rho_s} ;

        return inside_background && far_from_event_horizon;
    };

    // Pathtrace until a collision happens
    camera.pathtrace(collision_checker, dlam, metric);

    std::cout << "Writing to file!" << '\n';

    /*
        WRITE TO FILE
    */

    // ppm header
    filestream << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    // we iterate over each pixel and set its color accordingly
    for (int i {0}; i < image_height; ++i) {
        for (int j {0}; j < image_width; ++j) {

            // Get collision position
            Vec3 collision_pos = camera.get_paths()[i][j].get_position().get_vec3();
            Vec3 spherical_collision_pos = CoordinateSystem3::Cartesian_to_Spherical(collision_pos);

            double rho = spherical_collision_pos.norm();
            double rho_s = 2.*black_hole_mass/4.;
            double radius = rho*pow((1 + rho_s/rho),2);

            Vec3 pos = {radius, spherical_collision_pos[1], spherical_collision_pos[2]};

            // Get the ray's color and save as pixel_color
            Vec3 pixel_color = background.get_color(pos); // NOTE what is passed depends on the metric of choice

            //std::clog << "collision for " << i << ' ' << j << " is at " << spherical_collision_pos 
            //    << " with color " << pixel_color << '\n';

            // Write color to output stream
            filestream << int(pixel_color[0]) << ' ' << int(pixel_color[1]) << ' ' << int(pixel_color[2]) << '\n';

        }
    }

    // Finished!
    std::clog << "\rDone.                           \n";

    filestream.close();
    configstream.close();

    return 0;
}