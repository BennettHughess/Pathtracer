#pragma once
#include "cuda_classes.cuh"
#include "metric.h"

// forward declare vec4 to use
class CudaVec4;

class CudaMetric {

    private:

        // Metric possesses a type, is used to construct the rest of the metric
        // note: we can use the normal metric type
        Metric::MetricType type;

        // Also needs parameters (for now, just mass)
        MetricParameters params;
    
    public:

        // Constructor
        __host__ __device__ CudaMetric(Metric::MetricType t, MetricParameters p) : type {t}, params {p} {}
        __host__ __device__ CudaMetric() : type {Metric::CartesianIsotropicSchwarzschildMetric}, params {} {}

        // Access function
        __host__ __device__ Metric::MetricType get_type() { return type; }

        // Get metric components at a point in spacetime
        //std::vector<double> get_components(const Vec4& position);

        // Computes second derivative of position and returns as a 4-vector
        __host__ __device__ CudaVec4 get_acceleration(const CudaVec4& position, const CudaVec4& velocity);

};

CudaMetric Metric_to_CudaMetric(Metric metric);