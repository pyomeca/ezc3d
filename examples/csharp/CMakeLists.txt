# Copy the generated DLL to the C# project directory after building
project(${EZC3D_NAME}_CSharpExample LANGUAGES CSharp)

add_custom_target(
    CopyCSharpExampleFiles ALL
    DEPENDS ${CSHARP_PROJECT_NAME} ${EZC3D_NAME}
)

# Paths
set(SWIG_GENERATED_OUTPUT_DIR "${CSHARP_BUILD_DIR}/${CSHARP_GENERATED_FOLDER_NAME}")
set(SWIG_LIBRARY_OUTPUT_DIR "${CSHARP_BUILD_DIR}/$<CONFIG>")
set(CSHARP_EXAMPLE_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Copy generated C# files
add_custom_command(TARGET CopyCSharpExampleFiles
    DEPENDS ${CSHARP_PROJECT_NAME} ${EZC3D_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SWIG_GENERATED_OUTPUT_DIR}"
        "${CSHARP_EXAMPLE_OUTPUT_DIR}/${CSHARP_GENERATED_FOLDER_NAME}"
    COMMENT "Copying SWIG-generated C# files..."
)

# Copy compiled library
add_custom_command(TARGET CopyCSharpExampleFiles
    DEPENDS ${CSHARP_PROJECT_NAME} ${EZC3D_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SWIG_LIBRARY_OUTPUT_DIR}/${CSHARP_PROJECT_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${CSHARP_EXAMPLE_OUTPUT_DIR}"
    COMMENT "Copying compiled library..."
)