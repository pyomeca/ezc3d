cmake_minimum_required(VERSION 2.8)

project(ezC3D_test)
add_executable(${PROJECT_NAME} "main.cpp")


# Libraries
find_library(EZ_C3D ezC3D ../Debug)
target_link_libraries(${PROJECT_NAME} PUBLIC ${EZ_C3D})

# # Set some variables
# set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/modules)
# find_package(PythonLibs REQUIRED)
# target_link_libraries(${PROJECT_NAME} PUBLIC ${PYTHON_LIBRARY})

# Headers
target_include_directories(${PROJECT_NAME} PRIVATE ../include)


