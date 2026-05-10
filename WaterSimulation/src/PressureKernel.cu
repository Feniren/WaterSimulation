#include "PressureKernel.cuh"

__global__ void ComputePressure(
	int TotalParticleCount,
	const float* ParticleDensityList,
	float* ParticlePressureList,
	float RestDensity,
	float PressureStiffness
){
	int ThreadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (ThreadID >= TotalParticleCount){
		return;
	}

	float Density = ParticleDensityList[ThreadID];

	float Pressure = PressureStiffness * (Density - RestDensity);
	
	ParticlePressureList[ThreadID] = fmaxf(Pressure, 0.0f);
}