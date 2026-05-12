#include "Camera.h"

#include <glm/gtc/matrix_transform.hpp>

Camera::Camera(glm::vec3 Position, glm::vec3 Up, float Yaw, float Pitch){
	this->Position = Position;
	Front = glm::vec3(0.0f, 0.0f, -1.0f);
	WorldUp = Up;

	this->Pitch = Pitch;
	this->Yaw = Yaw;

	MovementSpeed = 5.0f;
	MouseSensitivity = 0.1f;
	Zoom = 45.0f;

	UpdateCameraVectors();
}

glm::mat4 Camera::GetViewMatrix() const{
	return glm::lookAt(Position, Position + Front, Up);
}

void Camera::ProcessKeyboard(CameraMovement Direction, float DeltaTime){
	float Velocity = MovementSpeed * DeltaTime;

	if (Direction == CameraMovement::Forward){
		Position += Front * Velocity;
	}

	if (Direction == CameraMovement::Backward){
		Position -= Front * Velocity;
	}

	if (Direction == CameraMovement::Left){
		Position -= Right * Velocity;
	}

	if (Direction == CameraMovement::Right){
		Position += Right * Velocity;
	}
}

void Camera::ProcessMouseMovement(float XOffset, float YOffset, bool ConstrainPitch){
	XOffset *= MouseSensitivity;
	YOffset *= MouseSensitivity;

	Yaw += XOffset;
	Pitch += YOffset;

	if (ConstrainPitch){
		if (Pitch > 89.0f){
			Pitch = 89.0f;
		}

		if (Pitch < -89.0f){
			Pitch = -89.0f;
		}
	}

	UpdateCameraVectors();
}

void Camera::UpdateCameraVectors(){
	glm::vec3 front;

	front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
	front.y = sin(glm::radians(Pitch));
	front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));

	Front = glm::normalize(front);

	Right = glm::normalize(glm::cross(Front, WorldUp));
	Up = glm::normalize(glm::cross(Right, Front));
}