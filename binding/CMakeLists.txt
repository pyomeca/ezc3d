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
