#pragma once

#include "Simulation.cuh"

void TestCUDATiming(WaterSimulation Simulation, int TimeStepCount);

void TestCPUTiming(WaterSimulation Simulation, int TimeStepCount);