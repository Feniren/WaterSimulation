#pragma once

#include <vector>

#include <cuda_runtime.h>

__global__ void ReorderParticles(
	int TotalParticleCount,
	const int* ParticleIndexList,
	const float3* ParticlePositionList,
	const float3* ParticleVelocityList,
	float3* SortedParticlePositionList,
	float3* SortedParticleVelocityList
);

void ReorderParticles(
	int TotalParticleCount,
	const std::vector<int>& ParticleIndexList,
	const std::vector<float3>& ParticlePositionList,
	const std::vector<float3>& ParticleVelocityList,
	std::vector<float3>& SortedParticlePositionList,
	std::vector<float3>& SortedParticleVelocityList
);