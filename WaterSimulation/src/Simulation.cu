#include <iostream>

//#include <cuda_gl_interop.h>

#include "Simulation.cuh"

#include "SimulationDebug.cuh"

#include "CellBoundsKernel.cuh"
#include "HashKernel.cuh"
#include "ParticleReorderKernel.cuh"
#include "NeighborSearchKernel.cuh"
#include "DensityKernel.cuh"
#include "PressureKernel.cuh"
#include "ForceKernel.cuh"
#include "IntegrationKernel.cuh"

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

	BoundaryDamping = -0.1f;

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

	GravityForce = make_float3(0.0f, -9.81f, 0.0f);

	TotalCellCount = (CellGridResolution.x * CellGridResolution.y * CellGridResolution.z);

	ParticleSpacing = (0.5f * SmoothingRadius);

	ParticleMass = (RestDensity * powf(ParticleSpacing, 3.0f));

	ParticleRadius = (0.5f * ParticleSpacing);

	Poly6Coefficient = (315.0f / (64.0f * 3.14159265358979323846f * powf(SmoothingRadius, 9.0f)));

	PressureStiffness = 200.0f;

	SpikyGradientCoefficient = (-45.0f / (3.14159265358979323846f * powf(SmoothingRadius, 6.0f)));

	TimeStep = 0.0002f;

	Viscosity = 0.1f;

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
}

WaterSimulation::~WaterSimulation(){
	shutdown();
	//delete impl;
}

void WaterSimulation::MakeGrid(){
	int Index = 0;

	for (int Z = 0; Z < ParticleGridZSize; Z++){
		for (int Y = 0; Y < ParticleGridYSize; Y++){
			for (int X = 0; X < ParticleGridXSize; X++){
				if (Index >= TotalParticleCount){
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

void WaterSimulation::InitializeSimulation(){
	MakeGrid();

	CUDA_CHECK(cudaMemcpy(DeviceParticleHashList, HostParticleHashList.data(), TotalParticleCount * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticleIndexList, HostParticleIndexList.data(), TotalParticleCount * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticleCellStartList, HostParticleCellStartList.data(), TotalCellCount * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticleCellEndList, HostParticleCellEndList.data(), TotalCellCount * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticleNeighborCountList, HostParticleNeighborCountList.data(), TotalParticleCount * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticlePositionList, HostParticlePositionList.data(), TotalParticleCount * sizeof(float3), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(DeviceParticleVelocityList, HostParticleVelocityList.data(), TotalParticleCount * sizeof(float3), cudaMemcpyHostToDevice));
}

/*bool WaterSimulation::init(const SimulationConfig& config) {
	impl->cfg = config;
	// allocate device buffers
	// create GL texture
	// register texture with CUDA
	return true;
}*/

void WaterSimulation::Step(bool DebugStep){
	int BlockSize = 256;
	int GridSize = (TotalParticleCount + BlockSize - 1) / BlockSize;

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

	if (DebugStep){
		CUDA_CHECK(cudaMemcpy(HostParticleHashList.data(), DeviceParticleHashList, TotalParticleCount * sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(HostParticleIndexList.data(), DeviceParticleIndexList, TotalParticleCount * sizeof(int), cudaMemcpyDeviceToHost));

		DebugSortParticleByHash(TotalParticleCount, HostParticleHashList, HostParticleIndexList);
	}

	CUDA_CHECK(cudaMemset(DeviceParticleCellStartList, 0xff, TotalCellCount * sizeof(int)));
	CUDA_CHECK(cudaMemset(DeviceParticleCellEndList, 0xff, TotalCellCount * sizeof(int)));

	FindCellStartEnd<<<GridSize, BlockSize>>>(TotalParticleCount, DeviceParticleHashList, DeviceParticleCellStartList, DeviceParticleCellEndList);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	if (DebugStep){
		CUDA_CHECK(cudaMemcpy(HostParticleCellStartList.data(), DeviceParticleCellStartList, TotalCellCount * sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(HostParticleCellEndList.data(), DeviceParticleCellEndList, TotalCellCount * sizeof(int), cudaMemcpyDeviceToHost));

		DebugFindCellBoundaries(TotalParticleCount, TotalCellCount, HostParticleCellStartList, HostParticleCellEndList);
	}

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

	if (DebugStep){
		CUDA_CHECK(cudaMemcpy(HostSortedParticlePositionList.data(), DeviceParticlePositionList, TotalParticleCount * sizeof(float3), cudaMemcpyDeviceToHost));

		DebugReorderParticles(TotalParticleCount, HostSortedParticlePositionList, HostParticleIndexList);
	}

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

	if (DebugStep){
		CUDA_CHECK(cudaMemcpy(HostParticleDensityList.data(), DeviceParticleDensityList, TotalParticleCount * sizeof(float), cudaMemcpyDeviceToHost));

		DebugComputeDensity(TotalParticleCount, HostParticleDensityList);
	}

	ComputePressure<<<GridSize, BlockSize>>>(TotalParticleCount, DeviceParticleDensityList, DeviceParticlePressureList, RestDensity, PressureStiffness);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	if (DebugStep){
		CUDA_CHECK(cudaMemcpy(HostParticlePressureList.data(), DeviceParticlePressureList, TotalParticleCount * sizeof(float), cudaMemcpyDeviceToHost));

		DebugComputePressure(TotalParticleCount, HostParticlePressureList, HostParticleDensityList);
	}

	ComputeForces<<<GridSize, BlockSize>>>(
		TotalParticleCount,
		DeviceParticlePositionList,
		DeviceParticleVelocityList,
		DeviceParticleDensityList,
		DeviceParticlePressureList,
		DeviceParticleForceList,
		DeviceParticleCellStartList,
		DeviceParticleCellEndList,
		BoxMin,
		CellSize,
		SmoothingRadius,
		ParticleMass,
		Viscosity,
		GravityForce,
		SpikyGradientCoefficient,
		ViscosityLaplacianCoefficient,
		CellGridResolution
	);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	if (DebugStep){
		CUDA_CHECK(cudaMemcpy(HostParticleForceList.data(), DeviceParticleForceList, TotalParticleCount * sizeof(float3), cudaMemcpyDeviceToHost));

		DebugComputeForces(TotalParticleCount, HostParticleForceList);
	}

	Integrate<<<GridSize, BlockSize>>>(
		TotalParticleCount,
		DeviceParticlePositionList,
		DeviceParticleVelocityList,
		DeviceParticleForceList,
		DeviceParticleDensityList,
		TimeStep,
		BoxMin,
		BoxMax,
		ParticleRadius,
		BoundaryDamping
	);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(HostParticlePositionList.data(), DeviceParticlePositionList, TotalParticleCount * sizeof(float3), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(HostParticleVelocityList.data(), DeviceParticleVelocityList, TotalParticleCount * sizeof(float3), cudaMemcpyDeviceToHost));

	if (DebugStep){
		DebugIntegrate(TotalParticleCount, HostParticlePositionList, HostParticleVelocityList);
	}
}

void WaterSimulation::StepSerial(bool DebugStep){

	ComputeParticleHashes(
		TotalParticleCount,
		HostParticlePositionList,
		HostParticleHashList,
		HostParticleIndexList,
		BoxMin,
		CellSize,
		CellGridResolution
	);

	SortParticlesByHash(TotalParticleCount, HostParticleHashList, HostParticleIndexList);

	if (DebugStep){
		DebugSortParticleByHash(TotalParticleCount, HostParticleHashList, HostParticleIndexList);
	}

	std::fill(HostParticleCellStartList.begin(), HostParticleCellStartList.end(), -1);
	std::fill(HostParticleCellEndList.begin(), HostParticleCellEndList.end(), -1);

	FindCellStartEnd(TotalParticleCount, TotalCellCount, HostParticleHashList, HostParticleCellStartList, HostParticleCellEndList);

	if (DebugStep){
		DebugFindCellBoundaries(TotalParticleCount, TotalCellCount, HostParticleCellStartList, HostParticleCellEndList);
	}

	ReorderParticles(
		TotalParticleCount,
		HostParticleIndexList,
		HostParticlePositionList,
		HostParticleVelocityList,
		HostSortedParticlePositionList,
		HostSortedParticleVelocityList
	);
	
	HostParticlePositionList.swap(HostSortedParticlePositionList);
	HostParticleVelocityList.swap(HostSortedParticleVelocityList);

	if (DebugStep){
		DebugReorderParticles(TotalParticleCount, HostSortedParticlePositionList, HostParticleIndexList);
	}

	ComputeDensity(
		TotalParticleCount,
		HostParticlePositionList,
		HostParticleDensityList,
		HostParticleCellStartList,
		HostParticleCellEndList,
		BoxMin,
		CellSize,
		SmoothingRadius,
		ParticleMass,
		Poly6Coefficient,
		CellGridResolution
	);

	if (DebugStep){
		DebugComputeDensity(TotalParticleCount, HostParticleDensityList);
	}

	ComputePressure(TotalParticleCount, HostParticleDensityList, HostParticlePressureList, RestDensity, PressureStiffness);

	if (DebugStep){
		DebugComputePressure(TotalParticleCount, HostParticlePressureList, HostParticleDensityList);
	}

	ComputeForces(
		TotalParticleCount,
		HostParticlePositionList,
		HostParticleVelocityList,
		HostParticleDensityList,
		HostParticlePressureList,
		HostParticleForceList,
		HostParticleCellStartList,
		HostParticleCellEndList,
		BoxMin,
		CellSize,
		SmoothingRadius,
		ParticleMass,
		Viscosity,
		GravityForce,
		SpikyGradientCoefficient,
		ViscosityLaplacianCoefficient,
		CellGridResolution
	);

	if (DebugStep){
		DebugComputeForces(TotalParticleCount, HostParticleForceList);
	}

	Integrate(
		TotalParticleCount,
		HostParticlePositionList,
		HostParticleVelocityList,
		HostParticleForceList,
		HostParticleDensityList,
		TimeStep,
		BoxMin,
		BoxMax,
		ParticleRadius,
		BoundaryDamping
	);

	if (DebugStep){
		DebugIntegrate(TotalParticleCount, HostParticlePositionList, HostParticleVelocityList);
	}
}

void WaterSimulation::inject(float x, float y, float radius, float amplitude) {
	// launch impulse kernel
}

const std::vector<float3> WaterSimulation::GetParticlePositionList(){
	return HostParticlePositionList;
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