# Copy the generated DLL to the C# project directory after building
project(${EZC3D_NAME}_CSharpExample)

add_custom_target(
    CopyCSharpExampleFiles ALL
    DEPENDS ${CSHARP_PROJECT_NAME} ${EZC3D_NAME}
)

# Paths
if (WIN32 AND MSVC)
    set(LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/$<CONFIG>")
    set(SWIG_LIBRARY_OUTPUT_DIR "${CSHARP_BUILD_DIR}/$<CONFIG>")
else()
    set(LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}")
    set(SWIG_LIBRARY_OUTPUT_DIR "${CSHARP_BUILD_DIR}")
endif()
set(SWIG_GENERATED_OUTPUT_DIR "${CSHARP_BUILD_DIR}/${CSHARP_GENERATED_FOLDER_NAME}")
set(CSHARP_EXAMPLE_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# lib prefix for linux and macos
if (UNIX)
    set(LIB_PREFIX "lib")
else()
    set(LIB_PREFIX "")
endif()

set(CSPROJ_NAME "ezc3dExample")
set(CSPROJ_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${CSPROJ_NAME}.csproj")
if (NOT EXISTS "${CSPROJ_PATH}")
    # Generate a project
    execute_process(
        COMMAND dotnet new console -o "${CMAKE_CURRENT_SOURCE_DIR}" --name "${CSPROJ_NAME}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE DOTNET_NEW_RESULT
        OUTPUT_VARIABLE DOTNET_NEW_OUTPUT
        ERROR_VARIABLE DOTNET_NEW_ERROR
    )
    if (NOT DOTNET_NEW_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to create C# example project using 'dotnet new console'. Please ensure that the .NET SDK is installed and available in your PATH.")
    endif()


    # Add the ezc3d library as a reference to the project
    # This is a hack and should be done properly at some point
    file(READ ${CSPROJ_PATH} CSPROJ_CONTENT)
    string(REPLACE "</Project>"
    "  <ItemGroup>
    <Content Include=\"\$(MSBuildProjectDirectory)\\**\\*\">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </Content>
  </ItemGroup>
</Project>"
    CSPROJ_CONTENT
    "${CSPROJ_CONTENT}")
    file(WRITE ${CSPROJ_PATH} "${CSPROJ_CONTENT}")

    # Remove the extra Program.cs file created by the dotnet new command
    file(REMOVE "${CMAKE_CURRENT_SOURCE_DIR}/Program.cs")
endif()
    
# Copy the core library to the example directory
if (EZC3D_SHARED)
    add_custom_command(TARGET CopyCSharpExampleFiles
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LIBRARY_OUTPUT_DIR}/${LIB_PREFIX}${EZC3D_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}"
            "${CSHARP_EXAMPLE_OUTPUT_DIR}"
        COMMENT "Copying compiled library to example directory..."
    )
endif()

# Copy generated C# files
add_custom_command(TARGET CopyCSharpExampleFiles POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SWIG_GENERATED_OUTPUT_DIR}"
        "${CSHARP_EXAMPLE_OUTPUT_DIR}/${CSHARP_GENERATED_FOLDER_NAME}"
    COMMENT "Copying SWIG-generated C# files..."
)

# Copy compiled library
add_custom_command(TARGET CopyCSharpExampleFiles POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SWIG_LIBRARY_OUTPUT_DIR}/${LIB_PREFIX}${CSHARP_PROJECT_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${CSHARP_EXAMPLE_OUTPUT_DIR}"
    COMMENT "Copying compiled library..."
)