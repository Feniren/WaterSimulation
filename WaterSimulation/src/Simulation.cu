#include <iostream>

//#include <cuda_gl_interop.h>

#include "Simulation.cuh"

#include "CellBoundsKernel.cuh"
#include "HashKernel.cuh"
#include "ParticleReorderKernel.cuh"
#include "NeighborSearchKernel.cuh"
#include "DensityKernel.cuh"
#include "PressureKernel.cuh"

#define CUDA_CHECK(Function) do { \
    cudaError_t Error = (Function); \
    \
    if (Error != cudaSuccess){ \
        printf("CUDA error %s:%d:\nCode: %d\nName: %s\nMessage: %s\n", __FILE__, __LINE__, (int)Error, cudaGetErrorName(Error), cudaGetErrorString(Error)); \
        exit(1); \
    } \
} while (0)

/*struct WaterSimulation::Impl {
	SimulationConfig cfg{};

	float* d_prev = nullptr;
	float* d_curr = nullptr;
	float* d_next = nullptr;

	GLuint heightTex = 0;
	cudaGraphicsResource* cudaHeightTex = nullptr;
};*/

WaterSimulation::WaterSimulation(int GridX, int GridY, int GridZ){
	ParticleGridXSize = GridX;
	ParticleGridYSize = GridY;
	ParticleGridZSize = GridZ;

	TotalParticleCount = (ParticleGridXSize * ParticleGridYSize * ParticleGridZSize);

	BoxMin = make_float3(0.0f, 0.0f, 0.0f);
	BoxMax = make_float3(2.0f, 2.0f, 2.0f);

	RestDensity = 1000.0f;

	SmoothingRadius = 0.04f;

	CellSize = SmoothingRadius;

	CellGridResolution = make_int3(
		(int)ceilf((BoxMax.x - BoxMin.x) / CellSize),
		(int)ceilf((BoxMax.y - BoxMin.y) / CellSize),
		(int)ceilf((BoxMax.z - BoxMin.z) / CellSize)
	);

	TotalCellCount = (CellGridResolution.x * CellGridResolution.y * CellGridResolution.z);

	ParticleSpacing = (0.5f * SmoothingRadius);

	ParticleMass = (RestDensity * powf(ParticleSpacing, 3.0f));

	Poly6Coefficient = (315.0f / (64.0f * 3.14159265358979323846f * powf(SmoothingRadius, 9.0f)));

	PressureStiffness = 200.0f;

	SpikyGradientCoefficient = (-45.0f / (3.14159265358979323846f * powf(SmoothingRadius, 6.0f)));

	ViscosityLaplacianCoefficient = (45.0f / (3.14159265358979323846f * powf(SmoothingRadius, 6.0f)));

	WaterStart = make_float3(0.2f, 0.2f, 0.2f);

	HostParticleHashList = std::vector<int>(TotalParticleCount);
	HostParticleIndexList = std::vector<int>(TotalParticleCount);
	HostParticleCellStartList = std::vector<int>(TotalCellCount, -1);
	HostParticleCellEndList = std::vector<int>(TotalCellCount, -1);
	HostParticleNeighborCountList = std::vector<int>(TotalCellCount);
	HostParticlePositionList = std::vector<float3>(TotalParticleCount);
	HostParticleVelocityList = std::vector<float3>(TotalParticleCount, make_float3(0.0f, 0.0f, 0.0f));
	HostSortedParticlePositionList = std::vector<float3>(TotalParticleCount);
	HostSortedParticleVelocityList = std::vector<float3>(TotalParticleCount, make_float3(0.0f, 0.0f, 0.0f));
	HostParticleForceList = std::vector<float3>(TotalParticleCount, make_float3(0.0f, 0.0f, 0.0f));
	HostParticleDensityList = std::vector<float>(TotalParticleCount, RestDensity);
	HostParticlePressureList = std::vector<float>(TotalParticleCount, 0.0f);
}

WaterSimulation::~WaterSimulation() {
	shutdown();
	//delete impl;
}

void WaterSimulation::MakeGrid(){
	int Index = 0;

	for (int Z = 0; Z < ParticleGridZSize; Z++){
		for (int Y = 0; Y < ParticleGridYSize; Y++){
			for (int X = 0; X < ParticleGridXSize; X++){
				if (Index >= TotalParticleCount) {
					break;
				}
				
				HostParticlePositionList[Index] = make_float3(
					WaterStart.x + X * ParticleSpacing,
					WaterStart.y + Y * ParticleSpacing,
					WaterStart.z + Z * ParticleSpacing
				);
				
				Index++;
			}
		}
	}
}

/*bool WaterSimulation::init(const SimulationConfig& config) {
	impl->cfg = config;
	// allocate device buffers
	// create GL texture
	// register texture with CUDA
	return true;
}*/

void WaterSimulation::step(){
	// launch update kernel
	// apply boundaries
	// swap prev/curr/next
	// map GL texture
	// write current height field into texture
	// unmap

	int* DeviceParticleHashList;
	int* DeviceParticleIndexList;
	int* DeviceParticleCellStartList;
	int* DeviceParticleCellEndList;
	int* DeviceParticleNeighborCountList;
	float3* DeviceParticlePositionList;
	float3* DeviceParticleVelocityList;
	float3* DeviceSortedParticlePositionList;
	float3* DeviceSortedParticleVelocityList;
	float3* DeviceParticleForceList;
	float* DeviceParticleDensityList;
	float* DeviceParticlePressureList;

	int BlockSize = 256;
	int GridSize = (TotalParticleCount + BlockSize - 1) / BlockSize;

	CUDA_CHECK(cudaMalloc((void**)&DeviceParticleHashList, TotalParticleCount * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&DeviceParticleIndexList, TotalParticleCount * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&DeviceParticleCellStartList, TotalCellCount * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&DeviceParticleCellEndList, TotalCellCount * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&DeviceParticleNeighborCountList, TotalCellCount * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&DeviceParticlePositionList, TotalParticleCount * sizeof(float3)));
	CUDA_CHECK(cudaMalloc((void**)&DeviceParticleVelocityList, TotalParticleCount * sizeof(float3)));
	CUDA_CHECK(cudaMalloc((void**)&DeviceSortedParticlePositionList, TotalParticleCount * sizeof(float3)));
	CUDA_CHECK(cudaMalloc((void**)&DeviceSortedParticleVelocityList, TotalParticleCount * sizeof(float3)));
	CUDA_CHECK(cudaMalloc(&DeviceParticleForceList, TotalParticleCount * sizeof(float3)));
	CUDA_CHECK(cudaMalloc(&DeviceParticleDensityList, TotalParticleCount * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&DeviceParticlePressureList, TotalParticleCount * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(DeviceParticleHashList, HostParticleHashList.data(), TotalParticleCount * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticleIndexList, HostParticleIndexList.data(), TotalParticleCount * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticleCellStartList, HostParticleCellStartList.data(), TotalCellCount * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticleCellEndList, HostParticleCellEndList.data(), TotalCellCount * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticleNeighborCountList, HostParticleNeighborCountList.data(), TotalCellCount * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticlePositionList, HostParticlePositionList.data(), TotalParticleCount * sizeof(float3), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticleVelocityList, HostParticleVelocityList.data(), TotalParticleCount * sizeof(float3), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceSortedParticlePositionList, HostSortedParticlePositionList.data(), TotalParticleCount * sizeof(float3), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceSortedParticleVelocityList, HostSortedParticleVelocityList.data(), TotalParticleCount * sizeof(float3), cudaMemcpyHostToDevice));

	ComputeParticleHashes<<<GridSize, BlockSize>>>(
		TotalParticleCount,
		DeviceParticlePositionList,
		DeviceParticleHashList,
		DeviceParticleIndexList,
		BoxMin,
		CellSize,
		CellGridResolution
	);

	CUDA_CHECK(cudaDeviceSynchronize());

	SortParticlesByHash(TotalParticleCount, DeviceParticleHashList, DeviceParticleIndexList);

	CUDA_CHECK(cudaMemcpy(HostParticleHashList.data(), DeviceParticleHashList, TotalParticleCount * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(HostParticleIndexList.data(), DeviceParticleIndexList, TotalParticleCount * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < TotalParticleCount; i++){
		if (i < 5){
			std::cout << "Sorted Particle " << i << " Hash = " << HostParticleHashList[i] << " Original Index = " << HostParticleIndexList[i] << std::endl;
		}

		if ((i > 0) && (HostParticleHashList[i] < HostParticleHashList[i - 1])){
			std::cout << "Sorting failed at " << i << std::endl;
		}
	}

	FindCellStartEnd<<<GridSize, BlockSize>>>(TotalParticleCount, DeviceParticleHashList, DeviceParticleCellStartList, DeviceParticleCellEndList);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(HostParticleCellStartList.data(), DeviceParticleCellStartList, TotalCellCount * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(HostParticleCellEndList.data(), DeviceParticleCellEndList, TotalCellCount * sizeof(int), cudaMemcpyDeviceToHost));

	int printed = 0;
	int countedParticles = 0;

	bool valid = true;

	for (int cell = 0; cell < TotalCellCount; cell++){
		int start = HostParticleCellStartList[cell];
		int end = HostParticleCellEndList[cell];

		if (start == -1 && end == -1) {
			continue;
		}

		if (start != -1){
			countedParticles += (end - start);

			if (printed < 5){
				std::cout << "Cell " << cell
					<< " Start = " << start
					<< " End = " << end
					<< " Count = " << (end - start)
					<< std::endl;

				printed++;
			}
		}

		if (start < 0 || end < 0 || start > end || end > TotalParticleCount){
			valid = false;

			std::cout << "Invalid bounds for cell " << cell
				<< " Start = " << start
				<< " End = " << end
				<< std::endl;

			break;
		}
	}

	if (valid){
		std::cout << "Cell start/end bounds look valid." << std::endl;
	}

	std::cout << "Counted particles from cells = "
		<< countedParticles
		<< " / "
		<< TotalParticleCount
		<< std::endl;

	ReorderParticles<<<GridSize, BlockSize>>>(
		TotalParticleCount,
		DeviceParticleIndexList,
		DeviceParticlePositionList,
		DeviceParticleVelocityList,
		DeviceSortedParticlePositionList,
		DeviceSortedParticleVelocityList
	);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	std::swap(DeviceParticlePositionList, DeviceSortedParticlePositionList);
	std::swap(DeviceParticleVelocityList, DeviceSortedParticleVelocityList);

	CUDA_CHECK(cudaMemcpy(HostSortedParticlePositionList.data(), DeviceParticlePositionList, TotalParticleCount * sizeof(float3), cudaMemcpyDeviceToHost));

	for (int i = 0; i < 2; i++){
		int OriginalIndex = HostParticleIndexList[i];
		float3 p = HostSortedParticlePositionList[i];

		std::cout << "Sorted slot " << i
			<< " Original Index = " << OriginalIndex
			<< " Pos = ("
			<< p.x << ", "
			<< p.y << ", "
			<< p.z << ")"
			<< std::endl;
	}

	CountNeighbors<<<GridSize, BlockSize>>>(
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

	ComputeDensity<<<GridSize, BlockSize>>>(
		TotalParticleCount,
		DeviceParticlePositionList,
		DeviceParticleDensityList,
		DeviceParticleCellStartList,
		DeviceParticleCellEndList,
		BoxMin,
		CellSize,
		SmoothingRadius,
		ParticleMass,
		Poly6Coefficient,
		CellGridResolution
	);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(HostParticleDensityList.data(), DeviceParticleDensityList, TotalParticleCount * sizeof(float), cudaMemcpyDeviceToHost));

	float minDensity = HostParticleDensityList[0];
	float maxDensity = HostParticleDensityList[0];
	double sumDensity = 0.0;

	for (int i = 0; i < TotalParticleCount; i++){
		float rho = HostParticleDensityList[i];

		minDensity = std::min(minDensity, rho);
		maxDensity = std::max(maxDensity, rho);
		sumDensity += rho;
	}

	double avgDensity = sumDensity / TotalParticleCount;

	std::cout << "Density min = " << minDensity << std::endl;
	std::cout << "Density max = " << maxDensity << std::endl;
	std::cout << "Density avg = " << avgDensity << std::endl;

	for (int i = 0; i < std::min(2, TotalParticleCount); i++){
		std::cout << "Particle " << i
			<< " Density = "
			<< HostParticleDensityList[i]
			<< std::endl;
	}

	ComputePressure<<<GridSize, BlockSize>>>(TotalParticleCount, DeviceParticleDensityList, DeviceParticlePressureList, RestDensity, PressureStiffness);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(HostParticlePressureList.data(), DeviceParticlePressureList, TotalParticleCount * sizeof(float), cudaMemcpyDeviceToHost));

	float minPressure = HostParticlePressureList[0];
	float maxPressure = HostParticlePressureList[0];
	double sumPressure = 0.0;
	int positiveCount = 0;

	for (int i = 0; i < TotalParticleCount; i++){
		float p = HostParticlePressureList[i];

		minPressure = std::min(minPressure, p);
		maxPressure = std::max(maxPressure, p);
		sumPressure += p;

		if (p > 0.0f){
			positiveCount++;
		}
	}

	double avgPressure = sumPressure / TotalParticleCount;

	std::cout << "Pressure min = " << minPressure << std::endl;
	std::cout << "Pressure max = " << maxPressure << std::endl;
	std::cout << "Pressure avg = " << avgPressure << std::endl;
	std::cout << "Positive pressure particles = "
		<< positiveCount
		<< " / "
		<< TotalParticleCount
		<< std::endl;

	for (int i = 0; i < std::min(20, TotalParticleCount); i++){
		std::cout << "Particle " << i
			<< " Density = " << HostParticleDensityList[i]
			<< " Pressure = " << HostParticlePressureList[i]
			<< std::endl;
	}
}

void WaterSimulation::inject(float x, float y, float radius, float amplitude) {
	// launch impulse kernel
}

/*GLuint WaterSimulation::getHeightTexture() const {
	return impl->heightTex;
}

int WaterSimulation::getWidth() const {
	return impl->cfg.width;
}

int WaterSimulation::getHeight() const {
	return impl->cfg.height;
}*/

void WaterSimulation::shutdown() {
	// unregister interop resource
	// delete texture
	// free device memory
}