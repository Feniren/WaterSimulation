#include <iostream>
#include <algorithm>

#include "HashKernel.cuh"

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__host__ __device__ int3 CalculateCellGridPosition(float3 ParticlePosition, float3 BoxMin, float CellSize){
	return make_int3(
		floorf((ParticlePosition.x - BoxMin.x) / CellSize),
		floorf((ParticlePosition.y - BoxMin.y) / CellSize),
		floorf((ParticlePosition.z - BoxMin.z) / CellSize)
	);
}

__host__ __device__ int CalculateCellGridHash(int3 CellPosition, int3 CellGridResolution, bool Clamp){
	if (Clamp){
		CellPosition = ClampCellGridPosition(CellPosition, CellGridResolution);
	}
	else{
		if (CellPosition.x < 0 || CellPosition.x >= CellGridResolution.x ||
			CellPosition.y < 0 || CellPosition.y >= CellGridResolution.y ||
			CellPosition.z < 0 || CellPosition.z >= CellGridResolution.z){
			return -1;
		}
	}

	return CellPosition.z * CellGridResolution.y * CellGridResolution.x
		+ CellPosition.y * CellGridResolution.x
		+ CellPosition.x;
}

__host__ __device__ int3 ClampCellGridPosition(int3 CellPosition, int3 CellGridResolution){
#if defined (__CUDA_ARCH__)
	CellPosition.x = max(0, min(CellPosition.x, CellGridResolution.x - 1));
	CellPosition.y = max(0, min(CellPosition.y, CellGridResolution.y - 1));
	CellPosition.z = max(0, min(CellPosition.z, CellGridResolution.z - 1));
#else
	CellPosition.x = std::max(0, std::min(CellPosition.x, CellGridResolution.x - 1));
	CellPosition.y = std::max(0, std::min(CellPosition.y, CellGridResolution.y - 1));
	CellPosition.z = std::max(0, std::min(CellPosition.z, CellGridResolution.z - 1));
#endif

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

	int Hash = CalculateCellGridHash(CellGridPosition, CellGridResolution, true);

	ParticleHashList[ThreadID] = Hash;
	ParticleIndexList[ThreadID] = ThreadID;
}

void ComputeParticleHashes(
	int TotalParticleCount,
	const std::vector<float3>& ParticlePositionList,
	std::vector<int>& ParticleHashList,
	std::vector<int>& ParticleIndexList,
	float3 BoxMin,
	float CellSize,
	int3 CellGridResolution
){
	for (int i = 0; i < TotalParticleCount; i++){
		int3 CellGridPosition = CalculateCellGridPosition(ParticlePositionList[i], BoxMin, CellSize);

		int Hash = CalculateCellGridHash(CellGridPosition, CellGridResolution, true);

		ParticleHashList[i] = Hash;
		ParticleIndexList[i] = i;
	}
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

void SortParticlesByHash(int TotalParticleCount, std::vector<int>& ParticleHashList, std::vector<int>& ParticleIndexList){
	std::vector<int> Order(TotalParticleCount);

	for (int i = 0; i < TotalParticleCount; i++){
		Order[i] = i;
	}

	std::sort(Order.begin(), Order.end(),
		[&](int a, int b)
		{
			return ParticleHashList[a] < ParticleHashList[b];
		}
	);

	std::vector<int> SortedHash(TotalParticleCount);
	std::vector<int> SortedIndex(TotalParticleCount);

	for (int i = 0; i < TotalParticleCount; i++){
		SortedHash[i] = ParticleHashList[Order[i]];
		SortedIndex[i] = ParticleIndexList[Order[i]];
	}

	ParticleHashList.swap(SortedHash);
	ParticleIndexList.swap(SortedIndex);
}