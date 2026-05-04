#pragma once

#include <GL/glew.h>

struct SimulationConfig {
	int width = 512;
	int height = 512;
	float dx = 1.0f;
	float dt = 0.1f;
	float waveSpeed = 1.2f;
	float damping = 0.015f;
};

class WaterSimulation {
public:
	WaterSimulation();
	~WaterSimulation();

	bool init(const SimulationConfig& config);
	void shutdown();

	void step();
	void inject(float x, float y, float radius, float amplitude);

	GLuint getHeightTexture() const;
	int getWidth() const;
	int getHeight() const;

private:
	struct Impl;
	Impl* impl;
};