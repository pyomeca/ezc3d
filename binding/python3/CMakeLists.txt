if(POLICY CMP0078)
    cmake_policy(SET CMP0078 NEW)
endif()
if(POLICY CMP0086)
    cmake_policy(SET CMP0086 OLD)
endif()

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13")
    project(_${EZC3D_NAME})
else()
    project(${EZC3D_NAME})
endif()

# Set up swig
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# include python
if(SKBUILD)
    find_package(NumPy REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIR})
    include_directories(${NumPy_INCLUDE_DIRS})
else()
    find_package(Python3 REQUIRED
        COMPONENTS Interpreter Development NumPy
    )
    include_directories(${Python3_INCLUDE_DIRS})
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
swig_link_libraries(${PROJECT_NAME}
    "${EZC3D_NAME}"
)

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.13")
    set(PYTHON_PROJECT_NAME ${PROJECT_NAME})
else()
    set(PYTHON_PROJECT_NAME _${PROJECT_NAME})
endif()

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
