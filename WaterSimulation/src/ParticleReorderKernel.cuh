#pragma once

#include <cuda_runtime.h>

__global__ void ReorderParticles(
	int TotalParticleCount,
	const int* ParticleIndexList,
	const float3* ParticlePositionList,
	const float3* ParticleVelocityList,
	float3* SortedParticlePositionList,
	float3* SortedParticleVelocityList
);