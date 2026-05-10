#pragma once

#include <cuda_runtime.h>

__global__ void ComputePressure(
	int TotalParticleCount,
	const float* ParticleDensityList,
	float* ParticlePressureList,
	float RestDensity,
	float PressureStiffness
);