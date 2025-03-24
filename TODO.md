# TO DO

## Near-term goals
- Implement image handling with PNG++ over PPMs (https://www.nongnu.org/pngpp/doc/0.2.1/)
- Update README.md
- Improve filesystem handling
    - Add preprocessor macros to hard code config.json location so main.exe can be run from anywhere

## Longer term goals
- Add a script to create video
- Optimize existing codebase
    - What if I passed everything as a float on the GPU side of things?
    - Better warp allocation on the GPU side? Profile the code.
- Restructure the Background and Camera classes to take input from the Scenario class and make it easier to develop code new code for different scenarios/metrics.

