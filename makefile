# Compiler
CXX = /opt/homebrew/Cellar/gcc/14.1.0_2/bin/g++-14

# Compiler flags
CXXFLAGS = -Wall

# Target executable
TARGET = main

# Functions
main: main.cpp vec3.cpp
	$(CXX) main.cpp vec3.cpp $(CXXFLAGS) -o $(TARGET) 