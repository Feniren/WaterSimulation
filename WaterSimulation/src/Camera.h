#pragma once

#include <glm/glm.hpp>

class Camera {
public:
	glm::vec3 Position;
	glm::vec3 Front;
	glm::vec3 Up;

	Camera(glm::vec3 position);

	glm::mat4 GetViewMatrix();
};