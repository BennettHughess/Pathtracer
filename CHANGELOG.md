Version 0.3.0

Changes:
    - Codebase has been completely retooled to trace photons in difference spacetimes
        - Currently supported spacetimes include Minkowski, Minkowski in spherical coordinates, and Schwarzschild
    - Introduced new classes: Vec4 and Metric. Both of these are used for pathtracing photons.
    - Coordinate conversions are now handled in the Vec3.cpp file.

Notes:
    - Speed of code has been dramatically reduced with all of the overhead added. Next steps are optimizing the
    code and implementing multiprocressing.