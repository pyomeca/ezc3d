cmake_minimum_required(VERSION 2.8)
set (Tutorial_VERSION_MAJOR 0)
set (Tutorial_VERSION_MINOR 1)

project(ezC3D)
add_library(${PROJECT_NAME} SHARED)

# Headers
target_include_directories(${PROJECT_NAME} PRIVATE include)

# Sources files
add_subdirectory(src include)

