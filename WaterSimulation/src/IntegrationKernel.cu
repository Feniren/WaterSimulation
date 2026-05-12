#include "IntegrationKernel.cuh"

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
){
	int ThreadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (ThreadID >= TotalParticleCount){
		return;
	}

	float Density = fmaxf(ParticleDensityList[ThreadID], 1e-6f);

	float3 Acceleration = make_float3(
		ParticleForceList[ThreadID].x / Density,
		ParticleForceList[ThreadID].y / Density,
		ParticleForceList[ThreadID].z / Density
	);

	ParticleVelocityList[ThreadID].x += Acceleration.x * TimeStep;
	ParticleVelocityList[ThreadID].y += Acceleration.y * TimeStep;
	ParticleVelocityList[ThreadID].z += Acceleration.z * TimeStep;

	ParticlePositionList[ThreadID].x += ParticleVelocityList[ThreadID].x * TimeStep;
	ParticlePositionList[ThreadID].y += ParticleVelocityList[ThreadID].y * TimeStep;
	ParticlePositionList[ThreadID].z += ParticleVelocityList[ThreadID].z * TimeStep;

	float3 MinimumBound = make_float3(
		BoxMin.x + ParticleRadius,
		BoxMin.y + ParticleRadius,
		BoxMin.z + ParticleRadius
	);

	float3 MaximumBound = make_float3(
		BoxMax.x - ParticleRadius,
		BoxMax.y - ParticleRadius,
		BoxMax.z - ParticleRadius
	);

	if (ParticlePositionList[ThreadID].x < MinimumBound.x){
		ParticlePositionList[ThreadID].x = MinimumBound.x;
		ParticleVelocityList[ThreadID].x *= BoundaryDamping;
	}

	if (ParticlePositionList[ThreadID].x > MaximumBound.x){
		ParticlePositionList[ThreadID].x = MaximumBound.x;
		ParticleVelocityList[ThreadID].x *= BoundaryDamping;
	}

	if (ParticlePositionList[ThreadID].y < MinimumBound.y){
		ParticlePositionList[ThreadID].y = MinimumBound.y;
		ParticleVelocityList[ThreadID].y *= BoundaryDamping;
	}

	if (ParticlePositionList[ThreadID].y > MaximumBound.y){
		ParticlePositionList[ThreadID].y = MaximumBound.y;
		ParticleVelocityList[ThreadID].y *= BoundaryDamping;
	}

	if (ParticlePositionList[ThreadID].z < MinimumBound.z){
		ParticlePositionList[ThreadID].z = MinimumBound.z;
		ParticleVelocityList[ThreadID].z *= BoundaryDamping;
	}

	if (ParticlePositionList[ThreadID].z > MaximumBound.z){
		ParticlePositionList[ThreadID].z = MaximumBound.z;
		ParticleVelocityList[ThreadID].z *= BoundaryDamping;
	}
}