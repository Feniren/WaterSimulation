#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <vector_types.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ int3 CalculateCellGridPosition(float3 ParticlePosition, float3 BoxMin, float CellSize);

__device__ int CalculateCellGridHash(int3 CellPosition, int3 CellGridResolution);

__device__ int3 ClampCellGridPosition(int3 CellPosition, int3 CellGridResolution);

__global__ void ComputeParticleHashes(int TotalParticleCount, const float3* ParticlePositionList, int* ParticleHashList, int* ParticleIndexList, float3 boxMin, float CellSize, int3 CellGridResolution);

void SortParticlesByHash(int TotalParticleCount, int* ParticleHashList, int* ParticleIndexList);