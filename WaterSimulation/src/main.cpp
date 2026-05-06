#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Simulation.cuh"

#include "Camera.h"

int main() {
	if (!glfwInit()) {
		std::cerr << "Failed to init GLFW\n";
		return -1;
	}

	GLFWwindow* window = glfwCreateWindow(800, 600, "WaterSimulation", nullptr, nullptr);

	if (!window) {
		std::cerr << "Failed to create window\n";

		glfwTerminate();

		return -1;
	}

	glfwMakeContextCurrent(window);

	if (glewInit() != GLEW_OK) {
		std::cerr << "Failed to init GLEW\n";

		glfwDestroyWindow(window);
		glfwTerminate();

		return -1;
	}

	glEnable(GL_DEPTH_TEST);

	Camera camera(glm::vec3(0.0f, 3.0f, 6.0f));
	
	WaterSimulation Sim(32, 32, 32);

	Sim.MakeGrid();
	Sim.step();

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glm::mat4 view = camera.GetViewMatrix();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}