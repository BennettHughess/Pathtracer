#pragma once
#include "cuda_classes.cuh"
#include "scenario.h"

/*
    I'm going to place routines in this file which are not used outside of CUDA, and which are not specifically
    part of a class (e.g. something in cuda_classes.cuh).
*/

// forward declare to avoid circular dependencies
class CudaPath;

__device__ bool cuda_collision_checker(CudaPath& path, Scenario::ScenarioType& scenario_type, ScenarioParameters& params);