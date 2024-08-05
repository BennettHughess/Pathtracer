Version 0.2.1 (part 2)

Updates:
    - Vec3 coordinate conversion is now handled in vec3.cpp
    - Added Vec4 class
    - Added Metric class
    - Retooled Camera and Path classes to handle paths which are 4-vectors
    - Camera now propagates Paths according to a metric in correspondence with the geodesic equation

Bug fixes:
    - Corrected mistake in the dot() product function.

Bugs introduced:
    - oh my god oh my god NOTHING WORKS nothing works oh my god
    - The image being produced is not close to right
    - Pathtracing needs to be looked at more closely