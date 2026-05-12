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
){
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

	for (int i = 0; i < TotalParticleCount; i++){
		float Density = std::max(ParticleDensityList[i], 1e-6f);

		float3 Acceleration = make_float3(
			ParticleForceList[i].x / Density,
			ParticleForceList[i].y / Density,
			ParticleForceList[i].z / Density
		);

		ParticleVelocityList[i].x += Acceleration.x * TimeStep;
		ParticleVelocityList[i].y += Acceleration.y * TimeStep;
		ParticleVelocityList[i].z += Acceleration.z * TimeStep;

		ParticlePositionList[i].x += ParticleVelocityList[i].x * TimeStep;
		ParticlePositionList[i].y += ParticleVelocityList[i].y * TimeStep;
		ParticlePositionList[i].z += ParticleVelocityList[i].z * TimeStep;

		if (ParticlePositionList[i].x < MinimumBound.x){
			ParticlePositionList[i].x = MinimumBound.x;
			ParticleVelocityList[i].x *= BoundaryDamping;
		}

		if (ParticlePositionList[i].x > MaximumBound.x){
			ParticlePositionList[i].x = MaximumBound.x;
			ParticleVelocityList[i].x *= BoundaryDamping;
		}

		if (ParticlePositionList[i].y < MinimumBound.y){
			ParticlePositionList[i].y = MinimumBound.y;
			ParticleVelocityList[i].y *= BoundaryDamping;
		}

		if (ParticlePositionList[i].y > MaximumBound.y){
			ParticlePositionList[i].y = MaximumBound.y;
			ParticleVelocityList[i].y *= BoundaryDamping;
		}

		if (ParticlePositionList[i].z < MinimumBound.z){
			ParticlePositionList[i].z = MinimumBound.z;
			ParticleVelocityList[i].z *= BoundaryDamping;
		}

		if (ParticlePositionList[i].z > MaximumBound.z){
			ParticlePositionList[i].z = MaximumBound.z;
			ParticleVelocityList[i].z *= BoundaryDamping;
		}
	}
}