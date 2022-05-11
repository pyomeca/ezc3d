if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13")
    cmake_policy(SET CMP0078 NEW)
endif()
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.14")
    cmake_policy(SET CMP0086 OLD)
endif()

# If at least one interface
if (BINDER_PYTHON3)
	# Set up swig
	find_package(SWIG REQUIRED)
	include(${SWIG_USE_FILE})
endif()

# Add subdirectories for each language if desired
if (BINDER_PYTHON3)
    add_subdirectory(python3)
endif()

if (BINDER_MATLAB)
    add_subdirectory(matlab)
endif()

if (BINDER_OCTAVE)
    add_subdirectory(octave)
endif()

