#pragma once

#include <vector>

#include <cuda_runtime.h>

__global__ void ComputePressure(
	int TotalParticleCount,
	const float* ParticleDensityList,
	float* ParticlePressureList,
	float RestDensity,
	float PressureStiffness
);

void ComputePressure(
	int TotalParticleCount,
	const std::vector<float>& ParticleDensityList,
	std::vector<float>& ParticlePressureList,
	float RestDensity,
	float PressureStiffness
);