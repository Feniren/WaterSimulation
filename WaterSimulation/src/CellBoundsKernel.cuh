#pragma once

#include <vector>

#include <cuda_runtime.h>

__global__ void FindCellStartEnd(int TotalParticleCount, const int* ParticleHashList, int* ParticleCellStartList, int* ParticleCellEndList);

void FindCellStartEnd(
	int TotalParticleCount,
	int TotalCellCount,
	const std::vector<int>& ParticleHashList,
	std::vector<int>& ParticleCellStartList,
	std::vector<int>& ParticleCellEndList
);