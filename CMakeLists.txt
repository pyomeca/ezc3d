project(ezc3d)
cmake_minimum_required(VERSION 3.8)

set (${PROJECT_NAME}_VERSION_MAJOR 0)
set (${PROJECT_NAME}_VERSION_MINOR 1)

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
aux_source_directory(src SRC_LIST)
set(BUILD_SHARED_LIBS TRUE CACHE BOOL "Choose if build should be a dynamic or static library")
if (BUILD_SHARED_LIBS)
	add_library(${PROJECT_NAME} SHARED ${SRC_LIST})
else()
	add_library(${PROJECT_NAME} STATIC ${SRC_LIST})
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_debug")

# Add headers
target_include_directories(${PROJECT_NAME} PUBLIC include)

# Installation
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
install(DIRECTORY include DESTINATION ${${PROJECT_NAME}_INCLUDE_FOLDER})
        
# Add binding subdirectory
add_subdirectory(binding)

# Add the example if asked
set(BUILD_EXAMPLE TRUE CACHE BOOL "Build a C++ example")
if (BUILD_EXAMPLE)
    add_subdirectory(example)
endif()







