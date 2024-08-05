# Pathtracer

Authors: Ben Hughes

This project is a pathtracer, which will eventually be retooled and repackaged into a general relativistic pathtracer - ideally, one which can generate images of a Schwarzschild black hole.

At the moment, the pathtracer is capable of pathtracing the interior of a sphere. The camera position, field of view, and other related quantities can be altered inside the `main.ccp` file.

To run the project, compile all of the .cpp files. Run the executable, and the output will be in a `main.ppm` file.

```console
ben@bens-mbp-4 Raytracer % ./main
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