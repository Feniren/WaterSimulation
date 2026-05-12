#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

layout (location = 3) in vec3 instancePosition;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoords;

uniform mat4 view;
uniform mat4 projection;

uniform float sphereScale;
uniform float instancePositionScale;

void main()
{
    vec3 localPosition = aPos * sphereScale;
    vec3 worldPosition = localPosition + (instancePosition * instancePositionScale);

    FragPos = worldPosition;
    Normal = normalize(aNormal);

    TexCoords = aTexCoords;

    gl_Position = projection * view * vec4(worldPosition, 1.0);
}