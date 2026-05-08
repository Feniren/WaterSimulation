#pragma once

#include <cuda_runtime.h>

__global__ void CountNeighbors(
	int TotalParticleCount,
	const float3* ParticlePositionList,
	const int* ParticleCellStartList,
	const int* ParticleCellEndList,
	int* ParticleNeighborCountList,
	float3 BoxMin,
	float CellSize,
	float SmoothingRadius,
	int3 CellGridResolution
);