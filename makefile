# Compiler
CXX := clang++
CUDACXX := nvcc

# Compiler flags
CXXFLAGS := -Wall -Werror -Wextra -Wpedantic -Wunused -Wshadow -c -O3 -fopenmp -std=c++17 -o
LDFLAGS := -Wall -Werror -Wextra -Wpedantic -Wunused -Wshadow -fopenmp -L/usr/local/cuda/lib64 -std=c++17 -lcudart -o
CUDACXX_FLAGS := -c -arch=sm_86 -ccbin=$(CXX) -O3 -dc -std=c++17 -o
CUDACXX_LFLAGS := -arch=sm_86 -ccbin=$(CXX) -O3 -dlink -std=c++17 -o

# Directories
SRC := src/
INC := include/
BLD := build/
TST := test/

# basic commands to use
all: bin/main bin/debug bin/cuda_debug

main: bin/main

debug: bin/debug

cuda_debug: bin/cuda_debug

clean:
	rm build/*; rm bin/*

# Compile tests
bin/main: $(BLD)background.o $(BLD)camera.o $(BLD)metric.o $(BLD)path.o $(BLD)vec3.o $(BLD)vec4.o $(BLD)main.o $(BLD)cuda_routines.o $(BLD)cuda_classes.o $(BLD)cuda_out.o
	$(CXX) $(BLD)main.o $(BLD)background.o $(BLD)camera.o $(BLD)metric.o $(BLD)path.o $(BLD)vec3.o $(BLD)vec4.o $(BLD)cuda_routines.o $(BLD)cuda_classes.o $(BLD)cuda_out.o $(LDFLAGS) bin/main

bin/debug: $(BLD)background.o $(BLD)camera.o $(BLD)metric.o $(BLD)path.o $(BLD)vec3.o $(BLD)vec4.o $(BLD)debug.o $(BLD)cuda_routines.o $(BLD)cuda_classes.o $(BLD)cuda_out.o
	$(CXX) $(BLD)debug.o $(BLD)background.o $(BLD)camera.o $(BLD)metric.o $(BLD)path.o $(BLD)vec3.o $(BLD)vec4.o $(BLD)cuda_routines.o $(BLD)cuda_classes.o $(BLD)cuda_out.o $(LDFLAGS) bin/debug

bin/cuda_debug: $(BLD)background.o $(BLD)camera.o $(BLD)metric.o $(BLD)path.o $(BLD)vec3.o $(BLD)vec4.o $(BLD)cuda_debug.o $(BLD)cuda_routines.o $(BLD)cuda_classes.o $(BLD)cuda_out.o
	$(CXX) $(BLD)cuda_debug.o $(BLD)background.o $(BLD)camera.o $(BLD)metric.o $(BLD)path.o $(BLD)vec3.o $(BLD)vec4.o $(BLD)cuda_routines.o $(BLD)cuda_classes.o $(BLD)cuda_out.o $(LDFLAGS) bin/cuda_debug

# Compile test objects
build/main.o: $(TST)main.cpp
	$(CXX) $(TST)main.cpp $(CXXFLAGS) $(BLD)main.o

build/debug.o: $(TST)debug.cpp
	$(CXX) $(TST)debug.cpp $(CXXFLAGS) $(BLD)debug.o

build/cuda_debug.o: $(TST)cuda_debug.cpp
	$(CXX) $(TST)cuda_debug.cpp $(CXXFLAGS) $(BLD)cuda_debug.o

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

# Compile cude device code (note: you need to use nvcc to link cuda code first, then link cuda_out.o later).
build/cuda_out.o: $(BLD)cuda_routines.o $(BLD)cuda_classes.o
	$(CUDACXX) $(BLD)cuda_routines.o $(BLD)cuda_classes.o $(CUDACXX_LFLAGS) $(BLD)cuda_out.o

# Compile other cuda objects
build/cuda_routines.o: $(SRC)cuda_routines.cu $(INC)cuda_routines.h
	$(CUDACXX) $(SRC)cuda_routines.cu $(CUDACXX_FLAGS) $(BLD)cuda_routines.o

build/cuda_classes.o: $(SRC)cuda_classes.cu $(INC)cuda_classes.cuh 
	$(CUDACXX) $(SRC)cuda_classes.cu $(CUDACXX_FLAGS) $(BLD)cuda_classes.o