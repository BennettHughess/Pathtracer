#include <string>
#include <vector>
#include "vec4.h"
#include "metric.h"
#include <iostream>

int main() {

    double pi = 3.14159;

    Metric metric { Metric::SchwarzschildMetric };

    Vec4 pos {0, 3, pi/2, 0};

    Vec4 vel {1, 0, 1, 0};

    std::cout << "acceleration for this thing is " << metric.get_acceleration(pos,vel) << '\n';

    return 0;
}