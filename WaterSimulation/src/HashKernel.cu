#include "HashKernel.cuh"

__device__ int3 CalculateCellGridPosition(float3 ParticlePosition, float3 BoxMin, float CellSize){
	return make_int3(
		floorf((ParticlePosition.x - BoxMin.x) / CellSize),
		floorf((ParticlePosition.y - BoxMin.y) / CellSize),
		floorf((ParticlePosition.z - BoxMin.z) / CellSize)
	);
}

__device__ int calcGridHash(int3 gridPos, int3 gridRes){
	// optionally wrap or clamp
	if (gridPos.x < 0 || gridPos.x >= gridRes.x ||
		gridPos.y < 0 || gridPos.y >= gridRes.y ||
		gridPos.z < 0 || gridPos.z >= gridRes.z) {
		return -1;
	}

	return gridPos.z * gridRes.y * gridRes.x +
		gridPos.y * gridRes.x +
		gridPos.x;
}

__global__ void computeHashes(int n, float3* pos, int* particleHash, int* particleIndex, float cellSize, int3 gridRes){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	int3 gridPos = calcGridPos(pos[i], cellSize);
	int hash = calcGridHash(gridPos, gridRes);

	particleHash[i] = hash;
	particleIndex[i] = i;
}