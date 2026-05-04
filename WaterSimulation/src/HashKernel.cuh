#include <cuda_runtime.h>
#include <vector_types.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

__device__ int3 calcGridPos(float3 p, float cellSize) {
	return make_int3(
		floorf(p.x / cellSize),
		floorf(p.y / cellSize),
		floorf(p.z / cellSize)
	);
}

__device__ int calcGridHash(int3 gridPos, int3 gridRes);

__global__ void computeHashes(int n, float3* pos, int* particleHash, int* particleIndex, float cellSize, int3 gridRes);