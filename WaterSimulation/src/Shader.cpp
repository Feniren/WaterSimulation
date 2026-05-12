#include "Shader.h"

#include <glad/glad.h>
#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <sstream>
#include <iostream>

Shader::Shader(const char* VertexPath, const char* FragmentPath){
	std::string vertexCode;
	std::string fragmentCode;

	std::ifstream vShaderFile(VertexPath);
	std::ifstream fShaderFile(FragmentPath);

	if (!vShaderFile.is_open()){
		std::cerr << "ERROR::SHADER::FILE_NOT_FOUND\n"
			<< "Could not open vertex shader file: "
			<< vertexPath << std::endl;
	}

	if (!fShaderFile.is_open()){
		std::cerr << "ERROR::SHADER::FILE_NOT_FOUND\n"
			<< "Could not open fragment shader file: "
			<< fragmentPath << std::endl;
	}

	std::stringstream vShaderStream;
	std::stringstream fShaderStream;

	vShaderStream << vShaderFile.rdbuf();
	fShaderStream << fShaderFile.rdbuf();

	vertexCode = vShaderStream.str();
	fragmentCode = fShaderStream.str();

	const char* vCode = vertexCode.c_str();
	const char* fCode = fragmentCode.c_str();

	unsigned int vertex;
	unsigned int fragment;

	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vCode, nullptr);
	glCompileShader(vertex);
	checkCompileErrors(vertex, "VERTEX");

	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fCode, nullptr);
	glCompileShader(fragment);
	checkCompileErrors(fragment, "FRAGMENT");

	ID = glCreateProgram();
	glAttachShader(ID, vertex);
	glAttachShader(ID, fragment);
	glLinkProgram(ID);
	checkCompileErrors(ID, "PROGRAM");

	glDeleteShader(vertex);
	glDeleteShader(fragment);
}

void Shader::Use(){
	glUseProgram(ID);
}

// uniform helpers

void Shader::SetBool(const std::string& Name, bool Value) const{
	glUniform1i(glGetUniformLocation(ID, Name.c_str()), (int)Value);
}

void Shader::SetInt(const std::string& Name, int Value) const{
	glUniform1i(glGetUniformLocation(ID, Name.c_str()), Value);
}

void Shader::SetFloat(const std::string& Name, float Value) const{
	glUniform1f(glGetUniformLocation(ID, Name.c_str()), Value);
}

void Shader::SetVec3(const std::string& Name, const glm::vec3& Value) const{
	glUniform3fv(glGetUniformLocation(ID, Name.c_str()), 1, &Value[0]);
}

void Shader::SetMat4(const std::string& Name, const glm::mat4& Mat) const{
	glUniformMatrix4fv(
		glGetUniformLocation(ID, Name.c_str()),
		1,
		GL_FALSE,
		glm::value_ptr(Mat)
	);
}

void Shader::CheckCompileErrors(unsigned int Shader, const std::string& Type){
	int success;
	char infoLog[1024];

	if (Type != "PROGRAM"){
		glGetShaderiv(Shader, GL_COMPILE_STATUS, &success);

		if (!success){
			glGetShaderInfoLog(Shader, 1024, nullptr, infoLog);
			std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: "
				<< Type << "\n"
				<< infoLog
				<< "\n----------------------------------------"
				<< std::endl;
		}
	}
	else{
		glGetProgramiv(Shader, GL_LINK_STATUS, &success);

		if (!success){
			glGetProgramInfoLog(Shader, 1024, nullptr, infoLog);
			std::cerr << "ERROR::PROGRAM_LINKING_ERROR of type: "
				<< Type << "\n"
				<< infoLog
				<< "\n----------------------------------------"
				<< std::endl;
		}
	}
}