# Compiler
CXX := clang++

# Compiler flags
#CXXFLAGS := -Wall -c -I /opt/homebrew/opt/llvm/include -fopenmp -std=c++17 -o
#LDFLAGS := -L /opt/homebrew/opt/llvm/lib -fopenmp -std=c++17 -o
CXXFLAGS := -Wall -c -O2 -fopenmp -std=c++17 -o
LDFLAGS := -fopenmp -std=c++17 -o

# Directories
SRC := src/
INC := include/
BLD := build/
TST := test/

# basic commands to use
all: bin/main bin/debug

main: bin/main

clean:
	rm build/*

# Compile tests
bin/main: $(BLD)background.o $(BLD)camera.o $(BLD)metric.o $(BLD)path.o $(BLD)vec3.o $(BLD)vec4.o $(BLD)main.o
	$(CXX) $(BLD)main.o $(BLD)background.o $(BLD)camera.o $(BLD)metric.o $(BLD)path.o $(BLD)vec3.o $(BLD)vec4.o $(LDFLAGS) bin/main

# Compile test objects
build/main.o: $(TST)main.cpp
	$(CXX) $(TST)main.cpp $(CXXFLAGS) $(BLD)main.o

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