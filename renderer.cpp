#include <iostream>

int main(){

    // Configure renderer
    
    const int image_width {1920};
    const int image_height {1080};

    // Save to file

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    // i,j are indices: i <-> columns, j <-> rows
    // we iterate over each pixel and set its color accordingly

    for (int i {0}; i < image_height; ++i) {
        for (int j {0}; j < image_width; ++j) {

            double r { double(i)/(image_height-1) };
            double g { double(j)/(image_width-1) };
            double b { double(255/2) };

            int ir { int(255*r) };
            int ig { int(255*g) };
            int ib { int(255*b) };

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
            
        }
    }

}