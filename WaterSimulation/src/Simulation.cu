#include "Simulation.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

struct WaterSimulation::Impl {
	SimulationConfig cfg{};

	float* d_prev = nullptr;
	float* d_curr = nullptr;
	float* d_next = nullptr;

	GLuint heightTex = 0;
	cudaGraphicsResource* cudaHeightTex = nullptr;
};

WaterSimulation::WaterSimulation() : impl(new Impl()) {}
WaterSimulation::~WaterSimulation() {
	shutdown();
	delete impl;
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