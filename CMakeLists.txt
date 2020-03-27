cmake_minimum_required(VERSION 3.8)
cmake_policy(SET CMP0048 NEW)
project(ezc3d VERSION 1.2.4)
set(CMAKE_CXX_STANDARD 11)

# Set some variables
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/modules)

# Set a default build type to 'Release' if none was specified
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	MESSAGE(STATUS "Setting build type to 'Release' as none was specified.")
	SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	# Set the possible values of build type for cmake-gui
	SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
ENDIF()

# Prepare add library
set(SRC_LIST
    src/Analogs.cpp
    src/Channel.cpp
    src/Data.cpp
    src/ezc3d.cpp
    src/Matrix.cpp
    src/Vector3d.cpp
    src/Frame.cpp
    src/Group.cpp
    src/Header.cpp
    src/Parameter.cpp
    src/Parameters.cpp
    src/Point.cpp
    src/Points.cpp
    src/Subframe.cpp
    src/modules/ForcePlatforms.cpp
)

set(BUILD_SHARED_LIBS TRUE CACHE BOOL "Choose if build should be a dynamic or static library")
if (BUILD_SHARED_LIBS)
	add_library(${PROJECT_NAME} SHARED ${SRC_LIST})
else()
	add_library(${PROJECT_NAME} STATIC ${SRC_LIST})
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_debug")

# Remember the CMAKE_CURRENT_BINARY_DIR (important if ezc3d is used as submodule)
set(EZC3D_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Create the configuration header file associated with the asked options
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/ezc3dConfig.h.in ${EZC3D_BINARY_DIR}/include/ezc3dConfig.h)

# Add headers
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EZC3D_BINARY_DIR}/include
)

# Install target
if(WIN32)
    set(${PROJECT_NAME}_LIB_FOLDER Lib)
    set(${PROJECT_NAME}_BIN_FOLDER bin)
    set(${PROJECT_NAME}_INCLUDE_FOLDER include/${PROJECT_NAME})
else()
    set(${PROJECT_NAME}_LIB_FOLDER ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME})
    set(${PROJECT_NAME}_BIN_FOLDER ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME})
    set(${PROJECT_NAME}_INCLUDE_FOLDER ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
endif()
install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${${PROJECT_NAME}_LIB_FOLDER}
        RUNTIME DESTINATION ${${PROJECT_NAME}_BIN_FOLDER}
        LIBRARY DESTINATION ${${PROJECT_NAME}_LIB_FOLDER}
        )
install(DIRECTORY include/ DESTINATION ${${PROJECT_NAME}_INCLUDE_FOLDER})
install(FILES ${EZC3D_BINARY_DIR}/include/ezc3dConfig.h DESTINATION ${${PROJECT_NAME}_INCLUDE_FOLDER})

# Prepare share
include(CMakePackageConfigHelpers)
configure_package_config_file(
  share/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${PROJECT_NAME}_LIB_FOLDER/cmake
)
install(FILES 
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  DESTINATION ${${PROJECT_NAME}_LIB_FOLDER}/cmake
)

# uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${EZC3D_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${EZC3D_BINARY_DIR}/cmake_uninstall.cmake)
endif()

# Binders
add_subdirectory(binding)

# Examples
if(NOT WIN32)
	option(BUILD_EXAMPLE "Build a C++ example" ON)
	if (BUILD_EXAMPLE)
		add_subdirectory(example)
	endif()
endif(NOT WIN32)

# Doc
option(BUILD_DOC "Build documentation" OFF)
if (BUILD_DOC)
    add_subdirectory(doc)
endif (BUILD_DOC)

option(GET_OFFICIAL_DOCUMENTATION "Automatically download the C3D documentation" OFF)
if (GET_OFFICIAL_DOCUMENTATION)
    file(
        DOWNLOAD https://www.c3d.org/docs/C3D_User_Guide.pdf
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/C3D_User_Guide.pdf
        TIMEOUT 60 # seconds
        TLS_VERIFY ON
    )
endif()

# Testing
option(BUILD_TESTS "Build all tests." OFF) # Makes boolean 'test' available.
if (BUILD_TESTS)
    add_subdirectory(test)
endif()
