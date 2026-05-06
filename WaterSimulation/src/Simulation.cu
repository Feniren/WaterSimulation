#include <iostream>

//#include <cuda_gl_interop.h>

#include "Simulation.cuh"

#include "CellBoundsKernel.cuh"
#include "HashKernel.cuh"

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

	WaterStart = make_float3(0.2f, 0.2f, 0.2f);

	HostParticleHashList = std::vector<int>(TotalParticleCount);
	HostParticleIndexList = std::vector<int>(TotalParticleCount);
	HostParticleCellStartList = std::vector<int>(TotalParticleCount, -1);
	HostParticleCellEndList = std::vector<int>(TotalParticleCount, -1);
	HostParticlePositionList = std::vector<float3>(TotalParticleCount);
	HostParticleVelocityList = std::vector<float3>(TotalParticleCount, make_float3(0.0f, 0.0f, 0.0f));
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
	float3* DeviceParticlePositionList;
	float3* DeviceParticleVelocityList;
	float3* DeviceParticleForceList;
	float* DeviceParticleDensityList;
	float* DeviceParticlePressureList;

	int BlockSize = 256;
	int GridSize = (TotalParticleCount + BlockSize - 1) / BlockSize;

	CUDA_CHECK(cudaMalloc((void**)&DeviceParticleHashList, TotalParticleCount * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&DeviceParticleIndexList, TotalParticleCount * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&DeviceParticleCellStartList, TotalParticleCount * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&DeviceParticleCellEndList, TotalParticleCount * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&DeviceParticlePositionList, TotalParticleCount * sizeof(float3)));
	CUDA_CHECK(cudaMalloc(&DeviceParticleVelocityList, TotalParticleCount * sizeof(float3)));
	CUDA_CHECK(cudaMalloc(&DeviceParticleForceList, TotalParticleCount * sizeof(float3)));
	CUDA_CHECK(cudaMalloc(&DeviceParticleDensityList, TotalParticleCount * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&DeviceParticlePressureList, TotalParticleCount * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(DeviceParticleHashList, HostParticleHashList.data(), TotalParticleCount * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticleIndexList, HostParticleIndexList.data(), TotalParticleCount * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticleCellStartList, HostParticleCellStartList.data(), TotalParticleCount * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticleCellEndList, HostParticleCellEndList.data(), TotalParticleCount * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticlePositionList, HostParticlePositionList.data(), TotalParticleCount * sizeof(float3), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticleVelocityList, HostParticleVelocityList.data(), TotalParticleCount * sizeof(float3), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticleForceList, HostParticleForceList.data(), TotalParticleCount * sizeof(float3), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticleDensityList, HostParticleDensityList.data(), TotalParticleCount * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticlePressureList, HostParticlePressureList.data(), TotalParticleCount * sizeof(float), cudaMemcpyHostToDevice));

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
		if (i < 20){
			std::cout << "Sorted Particle " << i << " Hash = " << HostParticleHashList[i] << " Original Index = " << HostParticleIndexList[i] << std::endl;
		}

		if ((i > 0) && (HostParticleHashList[i] < HostParticleHashList[i - 1])){
			std::cout << "Sorting failed at " << i << std::endl;
		}
	}

	FindCellStartEnd<<<GridSize, BlockSize>>>(TotalParticleCount, DeviceParticleHashList, DeviceParticleCellStartList, DeviceParticleCellEndList);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
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