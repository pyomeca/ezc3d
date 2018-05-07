cmake_minimum_required(VERSION 2.8)
set (Tutorial_VERSION_MAJOR 0)
set (Tutorial_VERSION_MINOR 1)

project(ezC3D)
add_executable(${PROJECT_NAME} "main.cpp")

subdirs(include src)
