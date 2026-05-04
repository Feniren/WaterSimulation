#include "Simulation.cuh"

#include <cuda_gl_interop.h>

struct WaterSimulation::Impl {
	SimulationConfig cfg{};

	float* d_prev = nullptr;
	float* d_curr = nullptr;
	float* d_next = nullptr;

	GLuint heightTex = 0;
	cudaGraphicsResource* cudaHeightTex = nullptr;
};

WaterSimulation::WaterSimulation(int GridX, int GridY, int GridZ) : impl(new Impl()){
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

	HostPosition = std::vector<float3>(TotalParticleCount);
	HostVelocity = std::vector<float3>(TotalParticleCount, make_float3(0.0f, 0.0f, 0.0f));
	HostForce = std::vector<float3>(TotalParticleCount, make_float3(0.0f, 0.0f, 0.0f));
	HostDensity = std::vector<float>(TotalParticleCount, RestDensity);
	HostPressure = std::vector<float>(TotalParticleCount, 0.0f);
}

WaterSimulation::~WaterSimulation() {
	shutdown();
	delete impl;
}

void WaterSimulation::MakeGrid(){
	int Index = 0;

	float3 Start = make_float3(0.2f, 0.2f, 0.2f);

	for (int Z = 0; Z < ParticleGridZSize; Z++){
		for (int Y = 0; Y < ParticleGridYSize; Y++){
			for (int X = 0; X < ParticleGridXSize; X++){
				if (Index >= TotalParticleCount) {
					break;
				}
				
				HostPosition[Index] = make_float3(
					Start.x + X * ParticleSpacing,
					Start.y + Y * ParticleSpacing,
					Start.z + Z * ParticleSpacing
				);
				
				Index++;
			}
		}
	}
}

bool WaterSimulation::init(const SimulationConfig& config) {
	impl->cfg = config;
	// allocate device buffers
	// create GL texture
	// register texture with CUDA
	return true;
}

void WaterSimulation::step() {
	// launch update kernel
	// apply boundaries
	// swap prev/curr/next
	// map GL texture
	// write current height field into texture
	// unmap
}

void WaterSimulation::inject(float x, float y, float radius, float amplitude) {
	// launch impulse kernel
}

GLuint WaterSimulation::getHeightTexture() const {
	return impl->heightTex;
}

int WaterSimulation::getWidth() const {
	return impl->cfg.width;
}

int WaterSimulation::getHeight() const {
	return impl->cfg.height;
}

void WaterSimulation::shutdown() {
	// unregister interop resource
	// delete texture
	// free device memory
}