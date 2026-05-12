#pragma once

#include <vector>

#include <cuda_runtime.h>

void DebugSortParticleByHash(int TotalParticleCount, const std::vector<int>& ParticleHashList, const std::vector<int>& ParticleIndexList);

void DebugFindCellBoundaries(int TotalParticleCount, int TotalCellCount, const std::vector<int>& ParticleCellStartList, const std::vector<int>& ParticleCellEndList);

void DebugReorderParticles(int TotalParticleCount, const std::vector<float3>& SortedParticlePositionList, const std::vector<int>& ParticleIndexList);

void DebugComputeDensity(int TotalParticleCount, const std::vector<float>& ParticleDensityList);

void DebugComputePressure(int TotalParticleCount, const std::vector<float>& ParticlePressureList, const std::vector<float>& ParticleDensityList);

void DebugComputeForces(int TotalParticleCount, const std::vector<float3>& ParticleForceList);

void DebugIntegrate(int TotalParticleCount, const std::vector<float3>& ParticlePositionList, const std::vector<float3>& ParticleVelocityList);

/*
* CountNeighbors<<<GridSize, BlockSize>>>(
		TotalParticleCount,
		DeviceParticlePositionList,
		DeviceParticleCellStartList,
		DeviceParticleCellEndList,
		DeviceParticleNeighborCountList,
		BoxMin,
		CellSize,
		SmoothingRadius,
		CellGridResolution
	);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(HostParticleNeighborCountList.data(), DeviceParticleNeighborCountList, TotalParticleCount * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < 2; i++){
		std::cout << "Particle " << i
			<< " Neighbor Count = "
			<< HostParticleNeighborCountList[i]
			<< std::endl;
	}

	int minCount = HostParticleNeighborCountList[0];
	int maxCount = HostParticleNeighborCountList[0];
	long long sumCount = 0;

	for (int i = 0; i < TotalParticleCount; i++){
		int c = HostParticleNeighborCountList[i];

		minCount = std::min(minCount, c);
		maxCount = std::max(maxCount, c);
		sumCount += c;
	}

	double avgCount = static_cast<double>(sumCount) / TotalParticleCount;

	std::cout << "Neighbor count min = " << minCount << std::endl;
	std::cout << "Neighbor count max = " << maxCount << std::endl;
	std::cout << "Neighbor count avg = " << avgCount << std::endl;
*/