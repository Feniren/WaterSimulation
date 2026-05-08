#include "ParticleReorderKernel.cuh"

__global__ void ReorderParticles(
	int TotalParticleCount,
	const int* ParticleIndexList,
	const float3* ParticlePositionList,
	const float3* ParticleVelocityList,
	float3* SortedParticlePositionList,
	float3* SortedParticleVelocityList
){
	int ThreadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (ThreadID >= TotalParticleCount){
		return;
	}

	int OriginalIndex = ParticleIndexList[ThreadID];

	SortedParticlePositionList[ThreadID] = ParticlePositionList[OriginalIndex];
	SortedParticleVelocityList[ThreadID] = ParticleVelocityList[OriginalIndex];
}