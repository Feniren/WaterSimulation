#include "CellBoundsKernel.cuh"

__global__ void FindCellStartEnd(int TotalParticleCount, const int* ParticleHashList, int* ParticleCellStartList, int* ParticleCellEndList){
	int ThreadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (ThreadID >= TotalParticleCount){
		return;
	}

	int CurrentHash = ParticleHashList[ThreadID];

	if (ThreadID == 0){
		ParticleCellStartList[CurrentHash] = 0;
	}
	else{
		int PreviousHash = ParticleHashList[(ThreadID - 1)];

		if (CurrentHash != PreviousHash){
			ParticleCellStartList[CurrentHash] = ThreadID;
			ParticleCellEndList[PreviousHash] = ThreadID;
		}
	}

	if (ThreadID == (TotalParticleCount - 1)){
		ParticleCellEndList[CurrentHash] = TotalParticleCount;
	}
}

void FindCellStartEnd(
	int TotalParticleCount,
	int TotalCellCount,
	const std::vector<int>& ParticleHashList,
	std::vector<int>& ParticleCellStartList,
	std::vector<int>& ParticleCellEndList
){
	for (int i = 0; i < TotalParticleCount; i++){
		int CurrentHash = ParticleHashList[i];

		if (i == 0){
			ParticleCellStartList[CurrentHash] = 0;
		}
		else{
			int PreviousHash = ParticleHashList[i - 1];

			if (CurrentHash != PreviousHash){
				ParticleCellStartList[CurrentHash] = i;
				ParticleCellEndList[PreviousHash] = i;
			}
		}

		if (i == TotalParticleCount - 1){
			ParticleCellEndList[CurrentHash] = TotalParticleCount;
		}
	}
}