# Include python
find_package(PYTHON3 REQUIRED)
mark_as_advanced(CLEAR Python3_EXECUTABLE)

include_directories(${Python3_INCLUDE_DIRS})

find_package(Numpy REQUIRED)

include_directories(${NUMPY_INCLUDES})
if(WIN32)
    link_directories(${Python3_RUNTIME_LIBRARY_DIRS}/libs)
endif()
# For some reasons, target_include_directories later in the file sometime fails
include_directories(
    ${EZC3D_BINARY_DIR}/include
)

# Add swig module
set(CMAKE_SWIG_FLAGS "")
set(I_SOURCE_FILE ${PROJECT_NAME}_python.i)

set_property(SOURCE ${I_SOURCE_FILE} PROPERTY CPLUSPLUS ON)
SWIG_ADD_LIBRARY(${PROJECT_NAME} TYPE MODULE LANGUAGE python SOURCES ${I_SOURCE_FILE})
swig_link_libraries(${PROJECT_NAME} ${PROJECT_NAME} ${PYTHON_LIBRARIES} ${NUMPY_LIBRARIES})

# Add headers
set_target_properties(_${PROJECT_NAME} PROPERTIES SWIG_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

if(APPLE)
    set_target_properties(_${PROJECT_NAME} PROPERTIES SUFFIX ".so")
    set_property(TARGET _${PROJECT_NAME} APPEND PROPERTY LINK_FLAGS "-flat_namespace -undefined suppress")
endif()
set_target_properties(_${PROJECT_NAME} PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${${PROJECT_NAME}_BIN_FOLDER}")
set_target_properties(_${PROJECT_NAME} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# Install to the proper directories
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.py 
    DESTINATION ${Python3_SITELIB}/${PROJECT_NAME}
)

INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
    DESTINATION ${Python3_SITELIB}/${PROJECT_NAME}
)


# Create the version file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in ${CMAKE_CURRENT_BINARY_DIR}/_version.py)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_version.py
    DESTINATION ${Python3_SITELIB}/${PROJECT_NAME}
)

# Check if swig has its version 3 or 4, to choose the right place to put the library
if(${SWIG_VERSION} MATCHES "^3(\.[0-9]*)*$")
    set(PYTHON_INSTALL_DESTINATION ${Python3_SITELIB})
elseif(${SWIG_VERSION} MATCHES "^4(\.[0-9]*)*$")
    set(PYTHON_INSTALL_DESTINATION ${Python3_SITELIB}/${PROJECT_NAME})
else()
    message( FATAL_ERROR "Swig must be version 3 or 4" )
endif()

# Install the library
INSTALL(TARGETS _${PROJECT_NAME}
    DESTINATION ${PYTHON_INSTALL_DESTINATION}
)
