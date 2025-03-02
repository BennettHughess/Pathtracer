# Compiler
#CXX := g++
CXX := nvcc
#CXX := clang++
CUDACXX := nvcc

# Compiler flags
#CXXFLAGS := -Wall -Werror -Wextra -Wpedantic -Wunused -Wshadow -c -O3 -fopenmp -std=c++17 -o
#LDFLAGS := -Wall -Werror -Wextra -Wpedantic -Wunused -Wshadow -fopenmp -std=c++17 -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" -lcudart -o
CXXFLAGS := -c -arch=sm_86 -O3 -std=c++17 -o
LDFLAGS := -std=c++17 -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" -lcudart -o
CUDACXXFLAGS := -c -arch=sm_86 -O3 -std=c++17 -o

# Directories
SRC := src/
INC := include/
BLD := build/
TST := test/

# basic commands to use
all: bin/main bin/debug

main: bin/main

debug: bin/debug

clean:
	rm build/*; rm bin/*

# Compile tests
bin/main: $(BLD)background.o $(BLD)camera.o $(BLD)metric.o $(BLD)path.o $(BLD)vec3.o $(BLD)vec4.o $(BLD)main.o $(BLD)cuda_routines.o
	$(CXX) $(BLD)main.o $(BLD)background.o $(BLD)camera.o $(BLD)metric.o $(BLD)path.o $(BLD)vec3.o $(BLD)vec4.o $(BLD)cuda_routines.o $(LDFLAGS) bin/main

bin/debug: $(BLD)background.o $(BLD)camera.o $(BLD)metric.o $(BLD)path.o $(BLD)vec3.o $(BLD)vec4.o $(BLD)debug.o $(BLD)cuda_routines.o
	$(CXX) $(BLD)debug.o $(BLD)background.o $(BLD)camera.o $(BLD)metric.o $(BLD)path.o $(BLD)vec3.o $(BLD)vec4.o $(BLD)cuda_routines.o $(LDFLAGS) bin/debug

# Compile test objects
build/main.o: $(TST)main.cpp
	$(CXX) $(TST)main.cpp $(CXXFLAGS) $(BLD)main.o

build/debug.o: $(TST)debug.cpp
	$(CXX) $(TST)debug.cpp $(CXXFLAGS) $(BLD)debug.o

# Compile source objects
build/background.o: $(SRC)background.cpp $(INC)background.h
	$(CXX) $(SRC)background.cpp $(CXXFLAGS) $(BLD)background.o

build/camera.o: $(SRC)camera.cpp $(INC)camera.h
	$(CXX) $(SRC)camera.cpp $(CXXFLAGS) $(BLD)camera.o

build/metric.o: $(SRC)metric.cpp $(INC)metric.h
	$(CXX) $(SRC)metric.cpp $(CXXFLAGS) $(BLD)metric.o

build/path.o: $(SRC)path.cpp $(INC)path.h
	$(CXX) $(SRC)path.cpp $(CXXFLAGS) $(BLD)path.o

build/vec3.o: $(SRC)vec3.cpp $(INC)vec3.h
	$(CXX) $(SRC)vec3.cpp $(CXXFLAGS) $(BLD)vec3.o

build/vec4.o: $(SRC)vec4.cpp $(INC)vec4.h
	$(CXX) $(SRC)vec4.cpp $(CXXFLAGS) $(BLD)vec4.o

# Compile cuda objects
build/cuda_routines.o: $(SRC)cuda_routines.cu $(INC)cuda_routines.cuh
	$(CUDACXX) $(SRC)cuda_routines.cu $(CUDACXXFLAGS) $(BLD)cuda_routines.o