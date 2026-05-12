#pragma once

#include <vector>

#include <cuda_runtime.h>

__host__ __device__ float Poly6Kernel(float DistanceSquared, float SmoothingRadiusSquared, float Poly6Coefficient);

__global__ void ComputeDensity(
	int TotalParticleCount,
	const float3* ParticlePositionList,
	float* ParticleDensityList,
	const int* ParticleCellStartList,
	const int* ParticleCellEndList,
	float3 BoxMin,
	float CellSize,
	float SmoothingRadius,
	float ParticleMass,
	float Poly6Coefficient,
	int3 CellGridResolution
);

void ComputeDensity(
	int TotalParticleCount,
	const std::vector<float3>& ParticlePositionList,
	std::vector<float>& ParticleDensityList,
	const std::vector<int>& ParticleCellStartList,
	const std::vector<int>& ParticleCellEndList,
	float3 BoxMin,
	float CellSize,
	float SmoothingRadius,
	float ParticleMass,
	float Poly6Coefficient,
	int3 CellGridResolution
);