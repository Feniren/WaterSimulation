#include <cuda_runtime.h>
#include <vector>
#include <vector_types.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

__device__ int3 CalculateCellGridPosition(float3 ParticlePosition, float3 BoxMin, float CellSize);

__device__ int calcGridHash(int3 gridPos, int3 gridRes);

__global__ void computeHashes(int n, float3* pos, int* particleHash, int* particleIndex, float cellSize, int3 gridRes);