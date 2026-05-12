#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/gtc/matrix_transform.hpp>

#include "Simulation.cuh"

#include "Camera.h"
#include "Shader.h"

Camera CameraReference(glm::vec3(0.0f, 3.0f, 6.0f));

float DeltaTime = 0.0f;
float LastFrame = 0.0f;

float LastX = 400.0f;
float LastY = 300.0f;
bool FirstMouse = true;

void CreateSphereMesh(
	std::vector<Vertex>& vertices,
	std::vector<unsigned int>& indices,
	unsigned int xSegments = 32,
	unsigned int ySegments = 16
)
{
	vertices.clear();
	indices.clear();

	for (unsigned int y = 0; y <= ySegments; ++y){
		for (unsigned int x = 0; x <= xSegments; ++x){
			float xSegment = static_cast<float>(x) / static_cast<float>(xSegments);
			float ySegment = static_cast<float>(y) / static_cast<float>(ySegments);

			float xPos = std::cos(xSegment * 2.0f * glm::pi<float>()) *
				std::sin(ySegment * glm::pi<float>());

			float yPos = std::cos(ySegment * glm::pi<float>());

			float zPos = std::sin(xSegment * 2.0f * glm::pi<float>()) *
				std::sin(ySegment * glm::pi<float>());

			Vertex vertex;
			vertex.Position = glm::vec3(xPos, yPos, zPos);
			vertex.Normal = glm::normalize(glm::vec3(xPos, yPos, zPos));
			vertex.TexCoords = glm::vec2(xSegment, ySegment);

			vertices.push_back(vertex);
		}
	}

	for (unsigned int y = 0; y < ySegments; ++y){
		for (unsigned int x = 0; x < xSegments; ++x){
			unsigned int i0 = y * (xSegments + 1) + x;
			unsigned int i1 = (y + 1) * (xSegments + 1) + x;
			unsigned int i2 = (y + 1) * (xSegments + 1) + x + 1;
			unsigned int i3 = y * (xSegments + 1) + x + 1;

			indices.push_back(i0);
			indices.push_back(i1);
			indices.push_back(i2);

			indices.push_back(i0);
			indices.push_back(i2);
			indices.push_back(i3);
		}
	}
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height){
	glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window){
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS){
		glfwSetWindowShouldClose(window, true);
	}

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS){
		CameraReference.ProcessKeyboard(CameraMovement::Forward, DeltaTime);
	}

	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS){
		CameraReference.ProcessKeyboard(CameraMovement::Backward, DeltaTime);
	}

	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS){
		CameraReference.ProcessKeyboard(CameraMovement::Left, DeltaTime);
	}

	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS){
		CameraReference.ProcessKeyboard(CameraMovement::Right, DeltaTime);
	}
}

void mouse_callback(GLFWwindow* window, double XPosition, double YPosition){
	if (FirstMouse){
		LastX = static_cast<float>(XPosition);
		LastY = static_cast<float>(YPosition);
		FirstMouse = false;
	}

	float xoffset = static_cast<float>(XPosition) - LastX;
	float yoffset = LastY - static_cast<float>(YPosition);

	LastX = static_cast<float>(XPosition);
	LastY = static_cast<float>(YPosition);

	CameraReference.ProcessMouseMovement(xoffset, yoffset);
}

struct Vertex{
	glm::vec3 Position;
	glm::vec3 Normal;
	glm::vec2 TexCoords;
};

int main(){
	if (!glfwInit()){
		std::cerr << "Failed to init GLFW\n";
		return -1;
	}

	int ScreenWidth = 800;
	int ScreenHeight = 600;

	GLFWwindow* window = glfwCreateWindow(ScreenWidth, ScreenHeight, "WaterSimulation", nullptr, nullptr);

	if (!window){
		std::cerr << "Failed to create window\n";

		glfwTerminate();

		return -1;
	}

	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorPosCallback(window, mouse_callback);

	glfwMakeContextCurrent(window);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	if (glewInit() != GLEW_OK){
		std::cerr << "Failed to init GLEW\n";

		glfwDestroyWindow(window);
		glfwTerminate();

		return -1;
	}

	Shader ShaderReference("../../../src/VertexShader.vert", "../../../src/FragmentShader.frag");

	glEnable(GL_DEPTH_TEST);
	
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

	glm::vec3 lightPos(2.0f, 4.0f, 2.0f);
	glm::vec3 lightColor(1.0f, 1.0f, 1.0f);
	glm::vec3 objectColor(0.7f, 0.7f, 0.7f);

	CameraReference.Position = glm::vec3(0.0f, 3.0f, 6.0f);
	CameraReference.Front = glm::vec3(0.0f, -0.4f, -1.0f);
	CameraReference.Up = glm::vec3(0.0f, 1.0f, 0.0f);

	//Sphere

	unsigned int sphereVAO;
	unsigned int sphereVBO;
	unsigned int sphereEBO;
	unsigned int instancePositionVBO;

	std::vector<Vertex> sphereVertices;
	std::vector<unsigned int> sphereIndices;

	CreateSphereMesh(sphereVertices, sphereIndices, 32, 16);

	glGenVertexArrays(1, &sphereVAO);
	glGenBuffers(1, &sphereVBO);
	glGenBuffers(1, &sphereEBO);

	glBindVertexArray(sphereVAO);

	glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
	glBufferData(
		GL_ARRAY_BUFFER,
		sphereVertices.size() * sizeof(Vertex),
		sphereVertices.data(),
		GL_STATIC_DRAW
	);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
	glBufferData(
		GL_ELEMENT_ARRAY_BUFFER,
		sphereIndices.size() * sizeof(unsigned int),
		sphereIndices.data(),
		GL_STATIC_DRAW
	);

	// location 0: vertex position
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		sizeof(Vertex),
		(void*)offsetof(Vertex, Position)
	);

	// location 1: vertex normal
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(
		1,
		3,
		GL_FLOAT,
		GL_FALSE,
		sizeof(Vertex),
		(void*)offsetof(Vertex, Normal)
	);

	// location 2: texture coordinates
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(
		2,
		2,
		GL_FLOAT,
		GL_FALSE,
		sizeof(Vertex),
		(void*)offsetof(Vertex, TexCoords)
	);

	const unsigned int maxSphereCount = 10000;

	glGenBuffers(1, &instancePositionVBO);

	glBindBuffer(GL_ARRAY_BUFFER, instancePositionVBO);
	glBufferData(
		GL_ARRAY_BUFFER,
		maxSphereCount * sizeof(float3),
		nullptr,
		GL_DYNAMIC_DRAW
	);

	// location 3: per-instance sphere origin
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(
		3,
		3,
		GL_FLOAT,
		GL_FALSE,
		sizeof(float3),
		(void*)0
	);

	// This is what makes it per-instance instead of per-vertex
	glVertexAttribDivisor(3, 1);

	glBindVertexArray(0);

	while (!glfwWindowShouldClose(window)){
		float CurrentFrame = static_cast<float>(glfwGetTime());

		DeltaTime = CurrentFrame - LastFrame;
		LastFrame = CurrentFrame;

		processInput(window);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

		glm::mat4 Model = glm::mat4(1.0f);
		glm::mat4 View = CameraReference.GetViewMatrix();
		glm::mat4 Projection = glm::perspective(
			glm::radians(45.0f),
			aspectRatio,
			0.1f,
			100.0f
		);

		ShaderReference.Use();

		ShaderReference.SetMat4("model", Model);
		ShaderReference.SetMat4("view", View);
		ShaderReference.SetMat4("projection", Projection);

		ShaderReference.SetVec3("lightPos", lightPos);
		ShaderReference.SetVec3("lightColor", lightColor);
		ShaderReference.SetVec3("objectColor", objectColor);

		glBindVertexArray(planeVAO);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}