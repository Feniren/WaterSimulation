#pragma once

#include <cuda_runtime.h>

__device__ float3 AddFloat3(float3 a, float3 b);

__device__ float3 SubtractFloat3(float3 a, float3 b);

__device__ float3 MultiplyFloat3(float3 a, float b);

__device__ float DotProductFloat3(float3 a, float3 b);

__device__ float LengthFloat3(float3 a);

__device__ float SpikyGradientMagnitude(
	float r,
	float h,
	float SpikyGradientCoefficient
);

__device__ float ViscosityLaplacian(
	float r,
	float h,
	float ViscosityLaplacianCoefficient
);

__global__ void ComputeForces(
	int TotalParticleCount,
	const float3* ParticlePositionList,
	const float3* ParticleVelocityList,
	const float* ParticleDensityList,
	const float* ParticlePressureList,
	float3* ParticleForceList,
	const int* ParticleCellStartList,
	const int* ParticleCellEndList,
	float3 BoxMin,
	float CellSize,
	float SmoothingRadius,
	float ParticleMass,
	float Viscosity,
	float3 GravityForce,
	float SpikyGradientCoefficient,
	float ViscosityLaplacianCoefficient,
	int3 GridResolution
);