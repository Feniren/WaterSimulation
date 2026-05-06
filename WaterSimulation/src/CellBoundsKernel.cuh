#pragma once

#include <cuda_runtime.h>

__global__ void FindCellStartEnd(int TotalParticleCount, const int* ParticleHashList, int* ParticleCellStartList, int* ParticleCellEndList);