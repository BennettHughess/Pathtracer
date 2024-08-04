# Compiler
CXX = /opt/homebrew/Cellar/gcc/14.1.0_2/bin/g++-14

# Compiler flags
CXXFLAGS = -Wall

# Target executable
TARGET = main

# Functions
main: main.cpp vec3.cpp pathtracer.cpp
	$(CXX) main.cpp vec3.cpp pathtracer.cpp camera.cpp $(CXXFLAGS) -o $(TARGET) 

debug: debug.cpp vec3.cpp pathtracer.cpp
	$(CXX) debug.cpp vec3.cpp pathtracer.cpp camera.cpp $(CXXFLAGS) -o debug