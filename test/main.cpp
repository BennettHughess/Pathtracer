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
#include "../include/scenario.h"
#include "../lib/json.hpp"

// for convenience
using json = nlohmann::json;

int main(int argc, char *argv[]) {

    // get directory location
    #ifdef DIR_PATH        // this gets defined when the makefile is run
        std::filesystem::path dir_path = DIR_PATH;
    #else
        std::filesystem::path dir_path = std::filesystem::current_path()/"..";
    #endif

    // get config and image path
    std::filesystem::path config_path = dir_path/"config.json";
    std::filesystem::path image_path;

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

    // Get filename for image to be saved as
    try {
        if (argc == 1) {        // check if filename was inputted
            image_path=(dir_path/"main.png").string(); // if not, default output file to main.png
        } 
        else if (argc == 2) {
            std::string input_filename {argv[1]}; // if so, use the inputted filename
            image_path=(dir_path/input_filename).string();
        }
        else {
            throw 17;
        }
    }
    catch (int Err) {
        std::cerr << "ERROR " << Err << ": Invalid argument count: " << argc << std::endl;
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
        background.load_img(config["background"]["image_path"]);
    }
    catch (int Err) {
        std::cerr << "ERROR " << Err << ": Failure in background.load_img. " 
            << "Attempted to open " << config["background"]["image_path"] << " -- check config file." << std::endl;
        std::clog << "Unable to load specified background image. Defaulting to opening " << dir_path/"images/milky_way.jpg" << std::endl;
        
        try {
            background.load_img(dir_path/"images/milky_way.jpg");
        }
        catch (int Err) {
            std::cerr << "ERROR " << Err << ": Failure in background.load_img. " 
            << "Attempted to open " << config["background"]["image_path"] << ". Terminating program." << std::endl;
            return 1;
        }

    }
    
    // Configure viewport
    const double viewport_fov {config["camera"]["viewport"]["fov"]}; //1.815 rads is valorant fov, 104 degrees
    const double viewport_distance {config["camera"]["viewport"]["distance"]};
    camera.set_viewport_settings(viewport_fov, viewport_distance);

    // Initialize the scenario
    std::string str_scenario_type {config["scenario"]["type"]};
    ScenarioParameters scenario_params {
        config["scenario"]["black_hole_mass"],
        background_radius
    };

    Scenario::ScenarioType scenario_type;
    if (str_scenario_type == "SphericalMinkowski") {
        scenario_type = Scenario::ScenarioType::SphericalMinkowski;
    }
    else if (str_scenario_type == "CartesianMinkowski") {
        scenario_type = Scenario::ScenarioType::CartesianMinkowski;
    }
    else if (str_scenario_type == "Schwarzschild") {
        scenario_type = Scenario::ScenarioType::Schwarzschild;
    }
    else if (str_scenario_type == "CartesianSchwarzschild") {
        scenario_type = Scenario::ScenarioType::CartesianSchwarzschild;
    }
    else {
        std::cerr << "Config file: unknown metric type." <<'\n';
        scenario_type = Scenario::ScenarioType::CartesianMinkowski;
    }

    Scenario scenario(scenario_type, scenario_params);

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
    camera.initialize_paths(scenario.get_metric(), integrator, max_dlam, min_dlam, tolerance);

    // Pathtrace until a collision happens
    try {
    camera.pathtrace(scenario, dlam);
    } catch (int Err) {
        std::cerr << "ERROR " << Err << ": Pathtrace failed." << std::endl;
        return 1;
    }

    /*
        WRITE TO FILE
    */

    std::cout << "Writing to file: " << image_path << '\n';

    // Declare array of pixel colors
    std::vector<std::vector<Vec3>> array(image_height, std::vector<Vec3>(image_width));

    // Initialize array of pixel colors
    for (int i {0}; i < image_height; ++i) {
        for (int j {0}; j < image_width; ++j) {

            // Get collision position
            Vec3 pos = scenario.get_pixel_pos(camera.get_paths()[i][j]);

            // Get the ray's color and save as pixel_color
            Vec3 pixel_color = background.get_color(pos);

            // set pixel color
            array[i][j] = pixel_color;

        }
    }

    // Save image file
    background.save_png(image_path, array);

    // Finished!
    std::clog << "\rDone.                           \n";

    configstream.close();

    return 0;
}