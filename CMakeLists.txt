cmake_minimum_required(VERSION 3.8)
project(ezc3d VERSION 1.5.19)

option(BUILD_SHARED_LIBS "Choose if build should be a dynamic or static library" ON)
# Option to allow checking of size for matrix getter (slower but safer if activated)
option(USE_MATRIX_FAST_ACCESSOR
    "If fast accessor (without bound checks) should be use. Prevent SegFault from the user not activated, but is much slower (mostly for Debug purposes)"
    ON
)
option(BUILD_EXAMPLE "Build a C++ example" ON)
option(BUILD_DOC "Build documentation" OFF)
option(GET_OFFICIAL_DOCUMENTATION "Automatically download the C3D documentation" OFF)
option(BUILD_TESTS "Build all tests." OFF) # Makes boolean 'test' available.
option(BINDER_PYTHON3 "Build Python SWIG module" OFF)
option(BINDER_MATLAB "Build Matlab module" OFF)
option(BINDER_OCTAVE "Build Octave module" OFF)

set(EZC3D_NAME ${PROJECT_NAME})
set(EZC3D_ROOT_FOLDER ${PROJECT_SOURCE_DIR})

set(EZC3D_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(EZC3D_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(EZC3D_VERSION_PATCH ${PROJECT_VERSION_PATCH})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/modules)

# Set a default build type to 'Release' if none was specified
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    MESSAGE(STATUS "Setting build type to 'Release' as none was specified.")
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
ENDIF()

# Prepare add library
set(SRC_LIST
    src/math/Matrix.cpp
    src/math/Matrix33.cpp
    src/math/Matrix44.cpp
    src/math/Matrix66.cpp
    src/math/Vector3d.cpp
    src/math/Vector6d.cpp
    src/Analogs.cpp
    src/AnalogsInfo.cpp
    src/AnalogsSubframe.cpp
    src/Channel.cpp
    src/Data.cpp
    src/DataStartInfo.cpp
    src/ezc3d.cpp
    src/Frame.cpp
    src/Group.cpp
    src/Header.cpp
    src/Parameter.cpp
    src/Parameters.cpp
    src/Point.cpp
    src/Points.cpp
    src/PointsInfo.cpp
    src/Rotation.cpp
    src/Rotations.cpp
    src/RotationsInfo.cpp
    src/RotationsSubframe.cpp
    src/modules/ForcePlatforms.cpp
)

add_library(${PROJECT_NAME} ${SRC_LIST})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

set_target_properties(${PROJECT_NAME} PROPERTIES
    DEBUG_POSTFIX "_debug"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

if(NOT BUILD_SHARED_LIBS AND (BINDER_PYTHON3 OR BINDER_MATLAB OR BINDER_OCTAVE))
    # static ezc3d must be PIC for bindings (always shared)
    set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Remember the CMAKE_CURRENT_BINARY_DIR (important if ezc3d is used as submodule)
set(EZC3D_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Create the configuration header file associated with the asked options
if (BUILD_SHARED_LIBS)
    set(EZC3D_SHARED TRUE)
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/ezc3d/ezc3dConfig.h.in
    ${EZC3D_BINARY_DIR}/include/ezc3d/ezc3dConfig.h
)

# Add headers
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/ezc3d>
    $<BUILD_INTERFACE:${EZC3D_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${EZC3D_BINARY_DIR}/include/ezc3d>
)

# Install target
include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME} EXPORT ezc3dTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} ${CMAKE_INSTALL_INCLUDEDIR}/ezc3d
)
install(DIRECTORY include/ezc3d DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    # Exclude ezc3dConfig.h.in.
    PATTERN "*.h.in" EXCLUDE
)
install(FILES "${EZC3D_BINARY_DIR}/include/ezc3d/ezc3dConfig.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ezc3d
)

# Prepare share
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/share/ezc3dConfig.cmake.in
    ezc3dConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ezc3d
)
write_basic_package_version_file(
    ezc3dConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)
export(
    EXPORT ezc3dTargets
    FILE ezc3dTargets.cmake
)
install(
    EXPORT ezc3dTargets
    FILE ezc3dTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ezc3d
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ezc3dConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/ezc3dConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ezc3d
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
if (BUILD_EXAMPLE)
    add_subdirectory(examples)
endif()


# Doc
if (BUILD_DOC)
    add_subdirectory(doc)
endif (BUILD_DOC)

if (GET_OFFICIAL_DOCUMENTATION)
    file(
        DOWNLOAD https://www.c3d.org/docs/C3D_User_Guide.pdf
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/C3D_User_Guide.pdf
        TIMEOUT 60 # seconds
        TLS_VERIFY ON
    )
endif()

# Testing
if (BUILD_TESTS)
    add_subdirectory(test)
endif()
