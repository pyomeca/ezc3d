if(POLICY CMP0086)
    cmake_policy(SET CMP0086 NEW)
endif()
if(POLICY CMP0078)
    cmake_policy(SET CMP0078 NEW)
endif()
if(POLICY CMP0122)
    cmake_policy(SET CMP0122 NEW)
endif()

project(${EZC3D_NAME}_CSharp)
set(CSHARP_PROJECT_NAME "${PROJECT_NAME}" CACHE INTERNAL "Name of the C# swig interface project")

find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# Add swig module
set(CMAKE_SWIG_FLAGS -namespace Ezc3d)
set(I_SOURCE_FILE "${EZC3D_NAME}_csharp.i")

set_property(SOURCE "${I_SOURCE_FILE}"
    PROPERTY CPLUSPLUS ON
)

# Create an output directory for the generated C# files
set(CSHARP_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "Directory where the generated C# files will be created")
set(CSHARP_GENERATED_FOLDER_NAME "${EZC3D_NAME}_csharp_wrapper" CACHE INTERNAL "Name of the folder where the generated C# files will be created")
file(MAKE_DIRECTORY ${CSHARP_BUILD_DIR}/${CSHARP_GENERATED_FOLDER_NAME})
set(CMAKE_SWIG_OUTDIR ${CSHARP_BUILD_DIR}/${CSHARP_GENERATED_FOLDER_NAME}) 

# Generate the wrapper
SWIG_ADD_LIBRARY(${PROJECT_NAME}
    TYPE MODULE
    LANGUAGE "csharp"
    SOURCES "${I_SOURCE_FILE}"
)
swig_link_libraries(${PROJECT_NAME}
    "${EZC3D_NAME}"
)

# Add headers
set_target_properties(${PROJECT_NAME} PROPERTIES
    SWIG_INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include;${CMAKE_CURRENT_SOURCE_DIR}/../../include/ezc3d;${EZC3D_BINARY_DIR}/include;${EZC3D_BINARY_DIR}/include/ezc3d"
)
