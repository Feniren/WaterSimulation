#include <iostream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Simulation.cuh"

#include "Camera.h"

int main(){
	if (!glfwInit()){
		std::cerr << "Failed to init GLFW\n";
		return -1;
	}

	GLFWwindow* window = glfwCreateWindow(800, 600, "WaterSimulation", nullptr, nullptr);

	if (!window){
		std::cerr << "Failed to create window\n";

		glfwTerminate();

		return -1;
	}

	glfwMakeContextCurrent(window);

	if (glewInit() != GLEW_OK){
		std::cerr << "Failed to init GLEW\n";

		glfwDestroyWindow(window);
		glfwTerminate();

		return -1;
	}

	glEnable(GL_DEPTH_TEST);

	Camera CameraReference(glm::vec3(0.0f, 3.0f, 6.0f));
	
	WaterSimulation Sim(32, 32, 32);

	Sim.MakeGrid();

	for (int i = 0; i < 2; i++){
		if (i % 10 == 0){
			std::cout << "\n Step: " << i << std::endl;

			Sim.Step(true);
		}
		else{
			Sim.Step(false);
		}
	}

	float planeVertices[] = {
		// positions              // normals         // texcoords
		-5.0f, 0.0f, -5.0f,       0.0f, 1.0f, 0.0f,  0.0f, 0.0f,
		 5.0f, 0.0f, -5.0f,       0.0f, 1.0f, 0.0f,  1.0f, 0.0f,
		 5.0f, 0.0f,  5.0f,       0.0f, 1.0f, 0.0f,  1.0f, 1.0f,

		-5.0f, 0.0f, -5.0f,       0.0f, 1.0f, 0.0f,  0.0f, 0.0f,
		 5.0f, 0.0f,  5.0f,       0.0f, 1.0f, 0.0f,  1.0f, 1.0f,
		-5.0f, 0.0f,  5.0f,       0.0f, 1.0f, 0.0f,  0.0f, 1.0f
	};

	unsigned int planeVAO, planeVBO;
	glGenVertexArrays(1, &planeVAO);
	glGenBuffers(1, &planeVBO);

	glBindVertexArray(planeVAO);

	glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);

	// position
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	// normal
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	// texcoord
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);

	CameraReference.Position = glm::vec3(0.0f, 3.0f, 6.0f);
	CameraReference.Front = glm::vec3(0.0f, -0.4f, -1.0f);
	CameraReference.Up = glm::vec3(0.0f, 1.0f, 0.0f);

	while (!glfwWindowShouldClose(window)){
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glm::mat4 view = CameraReference.GetViewMatrix();

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}