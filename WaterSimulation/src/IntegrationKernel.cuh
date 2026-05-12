#pragma once

#include <vector>

#include <cuda_runtime.h>

__global__ void Integrate(
	int TotalParticleCount,
	float3* ParticlePositionList,
	float3* ParticleVelocityList,
	const float3* ParticleForceList,
	const float* ParticleDensityList,
	float TimeStep,
	float3 BoxMin,
	float3 BoxMax,
	float ParticleRadius,
	float BoundaryDamping
);

void Integrate(
	int TotalParticleCount,
	std::vector<float3>& ParticlePositionList,
	std::vector<float3>& ParticleVelocityList,
	const std::vector<float3>& ParticleForceList,
	const std::vector<float>& ParticleDensityList,
	float TimeStep,
	float3 BoxMin,
	float3 BoxMax,
	float ParticleRadius,
	float BoundaryDamping
);