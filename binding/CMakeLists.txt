
option(BUILD_PYTHON "Build Python SWIG module" OFF)
option(BUILD_MATLAB "Build Matlab module" OFF)

# If at least one interface
if (BUILD_PYTHON)
	# Set up swig
	find_package(SWIG REQUIRED)
	include(${SWIG_USE_FILE})
endif()

# Add subdirectories for each language if desired
if(BUILD_PYTHON)
    add_subdirectory(Python3)
endif()

if(BUILD_MATLAB)
    add_subdirectory(matlab)
endif()