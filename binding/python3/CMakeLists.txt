if(POLICY CMP0078)
    cmake_policy(SET CMP0078 NEW)
endif()

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13")
    project(_${EZC3D_NAME})
else()
    project(${EZC3D_NAME})
endif()

# Set up swig
if(DEFINED SWIG_EXECUTABLE AND NOT EXISTS "${SWIG_EXECUTABLE}")
    unset(SWIG_EXECUTABLE CACHE)
endif()
if(DEFINED SWIG_DIR AND NOT EXISTS "${SWIG_DIR}")
    unset(SWIG_DIR CACHE)
endif()
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# include python
set(Python3_FIND_VIRTUALENV FIRST)
set(Python3_FIND_STRATEGY LOCATION)
if(DEFINED ENV{CONDA_PREFIX} AND NOT DEFINED Python3_ROOT_DIR)
    set(Python3_ROOT_DIR "$ENV{CONDA_PREFIX}")
endif()
if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/python" AND NOT DEFINED Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
endif()

if(SKBUILD)
    # Newer CMake versions handle NumPy through FindPython3.
    # Keep a fallback to legacy FindNumPy for older toolchains.
    find_package(Python3 QUIET COMPONENTS Interpreter Development NumPy)
    if(Python3_Interpreter_FOUND AND Python3_Development_FOUND AND Python3_NumPy_FOUND)
        include_directories(${Python3_INCLUDE_DIRS})
        include_directories(${Python3_NumPy_INCLUDE_DIRS})
    else()
        find_package(NumPy REQUIRED)
        include_directories(${PYTHON_INCLUDE_DIR})
        include_directories(${NumPy_INCLUDE_DIRS})
    endif()
else()
    # On some recent CMake + Python toolchains (notably free-threaded builds),
    # Development may be reported missing even when NumPy is available.
    find_package(Python3 REQUIRED
        COMPONENTS Interpreter NumPy
    )
    find_package(Python3 QUIET COMPONENTS Development)

    if(Python3_Development_FOUND)
        include_directories(${Python3_INCLUDE_DIRS})
    else()
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('include'))"
            OUTPUT_VARIABLE PYTHON3_INCLUDE_DIR_FROM_INTERPRETER
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(PYTHON3_INCLUDE_DIR_FROM_INTERPRETER)
            include_directories(${PYTHON3_INCLUDE_DIR_FROM_INTERPRETER})
        else()
            message(FATAL_ERROR "Could not determine Python include directory. Install Python development headers or use a Python environment with dev files.")
        endif()
    endif()

    include_directories(${Python3_NumPy_INCLUDE_DIRS})
endif()

if(WIN32)
    if(SKBUILD)
        get_filename_component(PYTHON_LIB_DIR ${PYTHON_LIBRARY} DIRECTORY)
        link_directories(${PYTHON_LIB_DIR})
    else()
        link_directories(${Python3_RUNTIME_LIBRARY_DIRS}/libs)
    endif()
endif()

# Add swig module
set(CMAKE_SWIG_FLAGS "")
set(I_SOURCE_FILE "${EZC3D_NAME}_python.i")

set_property(SOURCE "${I_SOURCE_FILE}"
    PROPERTY CPLUSPLUS ON
)
SWIG_ADD_LIBRARY(${PROJECT_NAME}
    TYPE MODULE
    LANGUAGE "python"
    SOURCES "${I_SOURCE_FILE}"
)
if(TARGET ${PROJECT_NAME})
    set(PYTHON_PROJECT_NAME ${PROJECT_NAME})
elseif(TARGET _${PROJECT_NAME})
    set(PYTHON_PROJECT_NAME _${PROJECT_NAME})
else()
    message(FATAL_ERROR "Could not find generated SWIG target for ${PROJECT_NAME}")
endif()
target_link_libraries(${PYTHON_PROJECT_NAME} PRIVATE
    "${EZC3D_NAME}"
)

# Add headers
set_target_properties(${PYTHON_PROJECT_NAME} PROPERTIES
    SWIG_INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include;${CMAKE_CURRENT_SOURCE_DIR}/../../include/ezc3d;${EZC3D_BINARY_DIR}/include;${EZC3D_BINARY_DIR}/include/ezc3d"
)

if(APPLE)
    set_target_properties(${PYTHON_PROJECT_NAME} PROPERTIES
        SUFFIX ".so")
    set_property(TARGET ${PYTHON_PROJECT_NAME} APPEND PROPERTY
        LINK_FLAGS "-flat_namespace -undefined suppress")
endif()
set_target_properties(${PYTHON_PROJECT_NAME} PROPERTIES
    PREFIX ""
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Define the python install 
if(SKBUILD)
    set(PYTHON_INSTALL_PREFIX "ezc3d" CACHE PATH
    "Path where to install the python binder, default is $Python_LIB/site-package/ezc3d"
    )
else()
    set(PYTHON_INSTALL_PREFIX "${Python3_SITELIB}/${EZC3D_NAME}" CACHE PATH
        "Path where to install the python binder, default is $Python_LIB/site-package/ezc3d"
    )
endif()

# Install to the proper directories
INSTALL(FILES "${CMAKE_CURRENT_BINARY_DIR}/${EZC3D_NAME}.py"
    DESTINATION "${PYTHON_INSTALL_PREFIX}"
)

INSTALL(FILES "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
    DESTINATION "${PYTHON_INSTALL_PREFIX}"
)


# Create the version file
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
    DESTINATION "${PYTHON_INSTALL_PREFIX}"
)

# Check if swig has its version 3 or 4, to choose the right place to put the library
if(${SWIG_VERSION} MATCHES "^3(\.[0-9]*)*$")
    if ("${PYTHON_INSTALL_PREFIX}" STREQUAL "${Python3_SITELIB}/${EZC3D_NAME}")
        set(PYTHON_PROJECT_INSTALL_DESTINATION "${Python3_SITELIB}")
    else()
        set(PYTHON_PROJECT_INSTALL_DESTINATION "${PYTHON_INSTALL_PREFIX}")
    endif()
elseif(${SWIG_VERSION} MATCHES "^4(\.[0-9]*)*$")
    set(PYTHON_PROJECT_INSTALL_DESTINATION "${PYTHON_INSTALL_PREFIX}")
else()
    message( FATAL_ERROR "Swig must be version 3 or 4" )
endif()
# Install the library
INSTALL(TARGETS ${PYTHON_PROJECT_NAME}
    DESTINATION "${PYTHON_PROJECT_INSTALL_DESTINATION}"
)
