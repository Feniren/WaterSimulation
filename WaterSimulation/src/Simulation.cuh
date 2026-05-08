#pragma once

#include <vector>

#include <cuda_runtime.h>

//#include <GL/glew.h>

/*struct SimulationConfig {
	int width = 512;
	int height = 512;
	float dx = 1.0f;
	float dt = 0.1f;
	float waveSpeed = 1.2f;
	float damping = 0.015f;
};*/

class WaterSimulation {
public:
	WaterSimulation(int GridX, int GridY, int GridZ);
	~WaterSimulation();

	//bool init(const SimulationConfig& config);
	void shutdown();

	void MakeGrid();

	void step();
	void inject(float x, float y, float radius, float amplitude);

	//GLuint getHeightTexture() const;
	//int getWidth() const;
	//int getHeight() const;

private:
	float3 BoxMin;
	float3 BoxMax;
	int3 CellGridResolution;
	float CellSize;
	int ParticleGridXSize;
	int ParticleGridYSize;
	int ParticleGridZSize;
	float ParticleMass;
	float ParticleSpacing;
	float Poly6Coefficient;
	float RestDensity;
	float SmoothingRadius;
	int TotalCellCount;
	int TotalParticleCount;
	float3 WaterStart;

	std::vector<int> HostParticleHashList;
	std::vector<int> HostParticleIndexList;
	std::vector<int> HostParticleCellStartList;
	std::vector<int> HostParticleCellEndList;
	std::vector<int> HostParticleNeighborCountList;
	std::vector<float3> HostParticlePositionList;
	std::vector<float3> HostParticleVelocityList;
	std::vector<float3> HostSortedParticlePositionList;
	std::vector<float3> HostSortedParticleVelocityList;
	std::vector<float3> HostParticleForceList;
	std::vector<float>  HostParticleDensityList;
	std::vector<float>  HostParticlePressureList;

	struct Impl;
	Impl* impl;
};