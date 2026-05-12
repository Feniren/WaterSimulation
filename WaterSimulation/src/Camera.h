#pragma once

#include <glm/glm.hpp>

enum CameraMovement{
	Forward,
	Backward,
	Left,
	Right
};

class Camera{
public:
	glm::vec3 Position;
	glm::vec3 Front;
	glm::vec3 Up;
	glm::vec3 Right;
	glm::vec3 WorldUp;

	float Yaw;
	float Pitch;

	float MovementSpeed;
	float MouseSensitivity;
	float Zoom;

	Camera(
		glm::vec3 Position = glm::vec3(0.0f, 3.0f, 6.0f),
		glm::vec3 Up = glm::vec3(0.0f, 1.0f, 0.0f),
		float Yaw = -90.0f,
		float Pitch = -25.0f
	);

	glm::mat4 GetViewMatrix() const;

	void ProcessKeyboard(CameraMovement Direction, float DeltaTime);
	void ProcessMouseMovement(float XOffset, float YOffset, bool ConstrainPitch = true);

private:
	void UpdateCameraVectors();
};