#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image.h"
#include "../lib/stb_image_write.h"
#include <iostream>

int main() {
    
    /********************   read from file    ********************/
    char filename[] = "images/milky_way.jpg"; // Update with correct file path

    int width, height, comp;
    unsigned char* data = stbi_load(filename, &width, &height, &comp, 0);
    if (!data) {
        std::cerr << "stbi_load failed: " << stbi_failure_reason() << std::endl;
        return 1;
    }

    std::cout << "Image loaded successfully!\n";
    std::cout << "Width: " << width << ", Height: " << height << ", Components: " << comp << std::endl;

    // Choose pixel (i, j)
    int i = 100; // Row
    int j = 150; // Column

    if (i >= height || j >= width) {
        std::cerr << "Error: Pixel index out of bounds!" << std::endl;
        stbi_image_free(data);
        return 1;
    }

    // Calculate pixel index
    int index = (i * width + j) * comp;

    // Read RGB values
    unsigned char r = data[index];
    unsigned char g = data[index + 1];
    unsigned char b = data[index + 2];

    std::cout << "Pixel (" << i << ", " << j << ") - R: " << (int)r
              << " G: " << (int)g << " B: " << (int)b << std::endl;

    stbi_image_free(data);

    /********************   write to file    ********************/

    char write_filename[] = "test.png";

    width = 256;   // Image width
    height = 256;  // Image height
    const int channels = 3;  // RGB (3 channels per pixel)

    // Allocate memory for the image
    unsigned char* new_data = new unsigned char[width * height * channels];

    // Generate a horizontal gradient (black to white)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = (y * width + x) * channels;
            unsigned char color = static_cast<unsigned char>(x * 255 / width); // Gradient intensity
            new_data[index] = color;     // Red
            new_data[index + 1] = color; // Green
            new_data[index + 2] = color; // Blue
        }
    }

    // Save the gradient as a PNG
    const char* output_file = "gradient.png";
    if (!stbi_write_png(output_file, width, height, channels, new_data, width * channels)) {
        std::cerr << "Failed to write PNG file!" << std::endl;
        delete[] new_data;
        return 1;
    }

    std::cout << "Saved PNG file: " << output_file << std::endl;

    // Free memory
    delete[] new_data;

    return 0;
}
