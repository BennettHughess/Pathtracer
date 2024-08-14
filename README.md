# Pathtracer

Authors: Ben Hughes

This project is capable of pathtracing null geodesics and recovering images around selected spacetimes. At the moment, the program is capable of tracing flat space (in Cartesian and spherical coordinates) and in the Schwarzschild metric. The camera position, field of view, and other related quantities can be altered inside the config.json file contained in the root directory of the project.

To run the project, compile all of the .cpp files in the /src folder along with main.cpp in the /test folder. Run the executable, and the output will be in a `main.ppm` file.

```console
ben@bens-mbp-4 Pathtracer % ./main
```

You may optionally include the name of the output file:

```console
ben@bens-mbp-4 Raytracer % ./main output.ppm
```

The resulting `.ppm` file which is created is an image file which can be viewed. On MacOS, Preview is capable of viewing these files. On Windows, I know GIMP can open these files as well.

At the moment, the program is set up to accept a .ppm image as a background image for the sphere. You can use ImageMagick to obtain a .ppm version of any image you like. On the command line, the line is

```console
ben@bens-mbp-4 images % magick filename.jpg -compress none filename.ppm
```

Current files:

- `main.cpp`
    - Compile and run this to execute the main code.
- `vec3.h` and `vec3.cpp`
    - These files contains the `Vec3` class, which describes three-dimensional vectors.
- `path.h` and `path.cpp`
    - These files contain the `Path` class, which contain basic tools for pathtracing.
- `camera.cpp` and `camera.h`
    - These files contain the `Camera` class, which provides the code for generating a camera, viewport, and various related functions like rotations and pathtracing.
- `background.cpp` and `background.h`
    - These files contain the `Background` class, which provides the tools for manipulating the background and storing an image to it.
- `vec4.cpp` and `vec4.h`
    - These files contain the `Vec4` class, which contains the basic information required to describe four-vectors in general relativity.
- `metric.cpp` and `metric.h`
    - These files contain the `Metric` class, which are primarily used to pass relevant information to the pathtracer, such as computing the Christoffel symbols.

## Configuration file
The configuration file `config.json`, which is located in the root directory of the project, contains all of the variables which configure the pathtracer. Currently, all of the variables should be specified to ensure the pathtracer runs correctly.

I included an example config file below, along with comments (yes, comments don't exist in json, but this is just an example to help).

```json
{
    "camera": 
    {
        // camera position in xyz coordinates
        "position": [0,-5,0],
        // the direction the camera is looking in xyz coords
        "direction": [1,0,0],
        // the direction which is "up" from the perspective of the camera
        "up": [0,-1,0],
        // any further rotations of the camera: [pitch, yaw, roll] (in radians)
        "rotation": [0,0,0],    
        "image":
        {
            // output image width and height in pixels
            "width": 1080,
            "height": 1920
        },
        "viewport":
        {
            // camera field of view in radians
            "fov": 1.815,
            // distance from origin to viewport
            "distance": 1
        }
    },
    "background":
    {
        // radius of background sphere
        "radius": 30,
        // background type: may either be an image or layered
        "type": "Image",
        // path from executable to background image
        // (only used when background type is image)
        "image_path": "../images/milky_way_hres.ppm"
    },
    "metric":
    {
        // black hole mass (only used with a black hole metric)
        "black_hole_mass": 1,
        // type of metric (may be CartesianMinkowskimetric,
        // SphericalMinkowskiMetric, or SchwarzschildMetric)
        "type": "SchwarzschildMetric"
    },
    "integrator":
    {
        // type of integrator used to solve ODE (Euler, RK4, etc.)
        "type": "CashKarp",
        // initial step size
        "dlam": 0.001,
        // min & max dlams and tolerance
        // for use with adaptive step-size integrators
        "max_dlam": 1,
        "min_dlam": 1.0E-20,
        "tolerance": 1.0E-12
    }
}
```