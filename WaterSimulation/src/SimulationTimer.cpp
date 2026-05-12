#include <iostream>
#include <chrono>

#include "SimulationTimer.h"

void TestCUDATiming(WaterSimulation Simulation, int TimeStepCount){
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaEventRecord(start);

	for (int i = 0; i < TimeStepCount; i++){
		Simulation.Step(false);
	}

	cudaEventRecord(end);
	cudaEventSynchronize(end);

	float milliseconds = 0.0f;
	cudaEventElapsedTime(&milliseconds, start, end);

	std::cout << "GPU time: " << milliseconds << " ms for " << TimeStepCount << " time steps" << std::endl;
	std::cout << "GPU time per step: "
		<< milliseconds / TimeStepCount
		<< " ms" << std::endl;

	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

void TestCPUTiming(WaterSimulation Simulation, int TimeStepCount){
	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < TimeStepCount; i++){
		Simulation.StepSerial(false);
	}

	auto end = std::chrono::high_resolution_clock::now();

	double milliseconds =
		std::chrono::duration<double, std::milli>(end - start).count();

	std::cout << "CPU time: " << milliseconds << " ms for " << TimeStepCount << " time steps" << std::endl;
	std::cout << "CPU time per step: "
		<< milliseconds / TimeStepCount
		<< " ms" << std::endl;
}