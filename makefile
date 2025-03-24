################ SETTINGS #################
# Compilers
CXX := clang++
CUDACXX := nvcc

############### LOGIC #####################
# Check if the CUDA compiler (probably nvcc) is installed using literal magic
# if nvcc is installed, find the path to the cuda folder and export it
HAS_NVCC := $(shell command -v $(CUDACXX) >/dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(HAS_NVCC),1)
CUDA_LIB_PATH := $(shell dirname $(shell command -v $(CUDACXX)))/../lib64
endif

# Add some info about nvcc
ifeq ($(HAS_NVCC),1)
$(info NOTE: $(CUDACXX) was found. Compiling with CUDA dependencies. CUDA lib location: $(CUDA_LIB_PATH))
else
$(info NOTE: $(CUDACXX) was not found. Compiling without CUDA dependencies.)
endif

# Compiler flags
ifeq ($(HAS_NVCC),1)
LDFLAGS := -Wall -Werror -Wextra -Wpedantic -Wunused -Wshadow -fopenmp -std=c++17 -L$(CUDA_LIB_PATH) -lcudart
CXXFLAGS := -Wall -Werror -Wextra -Wpedantic -Wunused -Wshadow -c -O3 -fopenmp -std=c++17 -DCUDA_INSTALLED
else
LDFLAGS := -Wall -Werror -Wextra -Wpedantic -Wunused -Wshadow -fopenmp -std=c++17
CXXFLAGS := -Wall -Werror -Wextra -Wpedantic -Wunused -Wshadow -c -O3 -fopenmp -std=c++17
endif
CUDACXX_FLAGS := -c -arch=sm_86 -ccbin=$(CXX) -O3 -dc -std=c++17
CUDACXX_LFLAGS := -arch=sm_86 -ccbin=$(CXX) -O3 -dlink -std=c++17

# Directories
SRC := src
INC := include
BLD := build
TST := test
BIN := bin

# Set up bin, build folders if they dont exist
$(shell mkdir -p $(BIN) $(BLD))

# Source test files
TESTS := main.cpp debug.cpp cuda_debug.cpp
CPP_SRCS := background.cpp camera.cpp metric.cpp path.cpp vec3.cpp vec4.cpp scenario.cpp
CUDA_SRCS := cuda_routines.cu cuda_classes.cu cuda_metric.cu cuda_misc.cu

# Object files (only define cuda files if NVCC installed)
ifeq ($(HAS_NVCC),1)
CUDA_OBJS := $(addprefix $(BLD)/, $(CUDA_SRCS:.cu=.o))
CUDA_OUT := $(BLD)/cuda_out.o
else
CUDA_OBJS :=
CUDA_OUT :=
endif

CPP_OBJS := $(addprefix $(BLD)/, $(CPP_SRCS:.cpp=.o))
TEST_OBJS := $(addprefix $(BLD)/, $(TESTS:.cpp=.o))

# Basic commands to use
ifeq ($(HAS_NVCC),1)
all: $(BIN)/main $(BIN)/debug $(BIN)/cuda_debug
else
all: $(BIN)/main $(BIN)/debug
endif

main: $(BIN)/main

debug: $(BIN)/debug

ifeq ($(HAS_NVCC),1)
cuda_debug: $(BIN)/cuda_debug
endif

clean:
	rm -f build/*
	rm -f bin/*

# Compile executables
$(BIN)/main: $(CPP_OBJS) $(CUDA_OBJS) $(CUDA_OUT) $(BLD)/main.o 
	$(CXX) $^ $(LDFLAGS) -o $@

$(BIN)/debug: $(CPP_OBJS) $(CUDA_OBJS) $(CUDA_OUT) $(BLD)/debug.o 
	$(CXX) $^ $(LDFLAGS) -o $@

ifeq ($(HAS_NVCC),1)
$(BIN)/cuda_debug: $(CPP_OBJS) $(CUDA_OBJS) $(CUDA_OUT) $(BLD)/cuda_debug.o 
	$(CXX) $^ $(LDFLAGS) -o $@
endif

# Compile test objects
$(BLD)/%.o: $(TST)/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Compile normal cpp objects
$(BLD)/%.o: $(SRC)/%.cpp $(INC)/%.h
	$(CXX) $(CXXFLAGS) $< -o $@

# Only do CUDA stuff if NVCC is installed
ifeq ($(HAS_NVCC),1)

# Compile cuda_out (for linking)
$(CUDA_OUT): $(CUDA_OBJS)
	$(CUDACXX) $^ $(CUDACXX_LFLAGS) -o $@

# Compile CUDA objects (and specially compile cuda_routines)
$(BLD)/%.o: $(SRC)/%.cu $(INC)/%.cuh
	$(CUDACXX) $(CUDACXX_FLAGS) $< -o $@

$(BLD)/cuda_routines.o: $(SRC)/cuda_routines.cu $(INC)/cuda_routines.h
	$(CUDACXX) $(CUDACXX_FLAGS) $< -o $@

endif
