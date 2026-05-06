#include <iostream>

#include "HashKernel.cuh"

__device__ int3 CalculateCellGridPosition(float3 ParticlePosition, float3 BoxMin, float CellSize){
	return make_int3(
		floorf((ParticlePosition.x - BoxMin.x) / CellSize),
		floorf((ParticlePosition.y - BoxMin.y) / CellSize),
		floorf((ParticlePosition.z - BoxMin.z) / CellSize)
	);
}

__device__ int CalculateCellGridHash(int3 CellPosition, int3 CellGridResolution){
	CellPosition = ClampCellGridPosition(CellPosition, CellGridResolution);

	return CellPosition.z * CellGridResolution.y * CellGridResolution.x
		+ CellPosition.y * CellGridResolution.x
		+ CellPosition.x;
}

__device__ int3 ClampCellGridPosition(int3 CellPosition, int3 CellGridResolution){
	CellPosition.x = max(0, min(CellPosition.x, CellGridResolution.x - 1));
	CellPosition.y = max(0, min(CellPosition.y, CellGridResolution.y - 1));
	CellPosition.z = max(0, min(CellPosition.z, CellGridResolution.z - 1));

	return CellPosition;
}

__global__ void ComputeParticleHashes(
	int TotalParticleCount,
	const float3* ParticlePositionList,
	int* ParticleHashList,
	int* ParticleIndexList,
	float3 BoxMin,
	float CellSize,
	int3 CellGridResolution
){
	int ThreadID = blockIdx.x * blockDim.x + threadIdx.x;

	if (ThreadID >= TotalParticleCount){
		return;
	}

	float3 ParticlePosition = ParticlePositionList[ThreadID];

	int3 CellGridPosition = CalculateCellGridPosition(ParticlePosition, BoxMin, CellSize);

	int Hash = CalculateCellGridHash(CellGridPosition, CellGridResolution);

	ParticleHashList[ThreadID] = Hash;
	ParticleIndexList[ThreadID] = ThreadID;
}

void SortParticlesByHash(int TotalParticleCount, int* ParticleHashList, int* ParticleIndexList){
	try{
		thrust::device_ptr<int> HashBegin(ParticleHashList);
		thrust::device_ptr<int> HashEnd(ParticleHashList + TotalParticleCount);
		thrust::device_ptr<int> IndexBegin(ParticleIndexList);

		thrust::sort_by_key(
			thrust::device,
			HashBegin,
			HashEnd,
			IndexBegin
		);

		cudaError_t Error = cudaDeviceSynchronize();

		if (Error != cudaSuccess){
			std::cerr << "CUDA error after sort: " << cudaGetErrorString(Error) << std::endl;
		}
	}
	catch (thrust::system_error& e){
		std::cerr << "Thrust Error: " << e.what() << std::endl;

		throw;
	}
}