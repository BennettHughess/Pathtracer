Version 0.5.3

Changes:

- Fixed compilation bug when running on non-cuda capable devices
- Changed sprintf to snprintf in stb_image_write library
- Modified makefile to not include /lib headers (since it didnt make any sense)
- Current default compiler is back to clang++ 
    - Note: on my mac, g++ defaults to running Apple Clang++, which doesn't support -fopenmp. Therefore, I need to specifically run Homebrew's g++-14.