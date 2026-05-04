#include <cuda_runtime.h>

__global__ void ComputeDensity(int n, float3* Position, float* Density, float h, float Mass);