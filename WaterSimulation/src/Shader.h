#pragma once

#include <string>
#include <glm/glm.hpp>

class Shader{
public:
	unsigned int ID;

	Shader(const char* VertexPath, const char* FragmentPath);

	void Use();

	void SetBool(const std::string& Name, bool Value) const;
	void SetInt(const std::string& Name, int Value) const;
	void SetFloat(const std::string& Name, float Value) const;

	void SetVec3(const std::string& Name, const glm::vec3& Value) const;
	void SetMat4(const std::string& Name, const glm::mat4& Mat) const;

private:
	void CheckCompileErrors(unsigned int Shader, const std::string& Type);
};