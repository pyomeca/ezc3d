# Copy the generated DLL to the C# project directory after building
project(${EZC3D_NAME}_CSharpTest LANGUAGES CSharp)

add_custom_target(
    CopyCSharpTestFiles ALL
    DEPENDS ${CSHARP_PROJECT_NAME} ${EZC3D_NAME}
)

# Paths
set(LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/$<CONFIG>")
set(SWIG_GENERATED_OUTPUT_DIR "${CSHARP_BUILD_DIR}/${CSHARP_GENERATED_FOLDER_NAME}")
set(SWIG_LIBRARY_OUTPUT_DIR "${CSHARP_BUILD_DIR}/$<CONFIG>")
set(CSHARP_TEST_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/")
# Create the output directory if it doesn't exist
file(MAKE_DIRECTORY "${CSHARP_TEST_OUTPUT_DIR}")

# Copy the core library to the test directory
add_custom_command(TARGET CopyCSharpTestFiles
    DEPENDS ${EZC3D_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LIBRARY_OUTPUT_DIR}/${EZC3D_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${CSHARP_TEST_OUTPUT_DIR}"
    COMMENT "Copying compiled library to test directory..."
)

# Copy generated C# files
add_custom_command(TARGET CopyCSharpTestFiles
    DEPENDS ${CSHARP_PROJECT_NAME} ${EZC3D_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SWIG_GENERATED_OUTPUT_DIR}"
        "${CSHARP_TEST_OUTPUT_DIR}/${CSHARP_GENERATED_FOLDER_NAME}"
    COMMENT "Copying SWIG-generated C# files..."
)

# Copy compiled library
add_custom_command(TARGET CopyCSharpTestFiles
    DEPENDS ${CSHARP_PROJECT_NAME} ${EZC3D_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SWIG_LIBRARY_OUTPUT_DIR}/${CSHARP_PROJECT_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${CSHARP_TEST_OUTPUT_DIR}"
    COMMENT "Copying compiled library..."
)