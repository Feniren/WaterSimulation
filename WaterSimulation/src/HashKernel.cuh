#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <vector_types.h>

__device__ int3 CalculateCellGridPosition(float3 ParticlePosition, float3 BoxMin, float CellSize);

__device__ int CalculateCellGridHash(int3 CellPosition, int3 CellGridResolution, bool Clamp);

__device__ int3 ClampCellGridPosition(int3 CellPosition, int3 CellGridResolution);

__global__ void ComputeParticleHashes(int TotalParticleCount, const float3* ParticlePositionList, int* ParticleHashList, int* ParticleIndexList, float3 boxMin, float CellSize, int3 CellGridResolution);

void SortParticlesByHash(int TotalParticleCount, int* ParticleHashList, int* ParticleIndexList);