cmake_minimum_required(VERSION 3.24)
project(WaterSimulation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(WaterSimulation
	src/main.cpp
)

target_link_libraries(WaterSimulation PRIVATE
    glfw
    GLEW::GLEW
	glm::glm
    OpenGL::GL
)