#include "../lib/json.hpp"
#include "../include/vec3.h"
#include <fstream>
#include <iostream>
#include <vector>

using json = nlohmann::json;

int main() {
    std::ifstream configstream("../config.json");
    json config { json::parse(configstream) };
    config = config[0];

    Vec3 position {config["camera"]["position"].template get<std::vector<double>>() };
    std::cout<<position<<'\n';

    configstream.close();

    return 0;
}
