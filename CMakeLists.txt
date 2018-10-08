project(ezc3d)
cmake_minimum_required(VERSION 3.10)

set (${PROJECT_NAME}_VERSION_MAJOR 0)
set (${PROJECT_NAME}_VERSION_MINOR 2)

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
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

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
if(NOT WIN32)
	set(BUILD_EXAMPLE TRUE CACHE BOOL "Build a C++ example")
	if (BUILD_EXAMPLE)
		add_subdirectory(example)
	endif()
endif(NOT WIN32)



################################
# Testing
################################

# Options. Turn on with 'cmake -DBUILD_TESTS=ON'.
option(BUILD_TESTS "Build all tests." OFF) # Makes boolean 'test' available.

if (BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)

    ##############
    # Unit Tests
    ##############
    file(GLOB TEST_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp)
    add_executable(runUnitTests ${TEST_SRC_FILES})

    # Include the gtest library.
    target_include_directories(runUnitTests PUBLIC ${GTEST_INCLUDE_DIRS})

    # Standard linking to gtest stuff.
    target_link_libraries(runUnitTests ${GTEST_BOTH_LIBRARIES} pthread)

    # Extra linking for the project.
    target_link_libraries(runUnitTests ${PROJECT_NAME})

    # This is so you can do 'make test' to see all your tests run, instead of
    # manually running the executable runUnitTests to see those specific tests.
    add_test(UnitTests runUnitTests)

    if (CMAKE_BUILD_TYPE STREQUAL "Coverage")
        set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/.travis/cmake)

        include(CodeCoverage)
        setup_target_for_coverage(${PROJECT_NAME}_coverage runUnitTests coverage)

        SET(CMAKE_CXX_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
        SET(CMAKE_C_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
    endif() #CMAKE_BUILD_TYPE STREQUAL "Coverage"
endif()




