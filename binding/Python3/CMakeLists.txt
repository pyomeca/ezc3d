# Include python
find_package(PYTHON3 REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

set(CMAKE_SWIG_FLAGS "")
set_source_files_properties(../${PROJECT_NAME}.i PROPERTIES CPLUSPLUS ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)


# Add swig module
SWIG_ADD_LIBRARY(ezC3D LANGUAGE python SOURCES ../${PROJECT_NAME}.i)
swig_link_libraries(ezC3D ${PROJECT_NAME} ${PYTHON_LIBRARIES})

# Files to install with Python
set(PYTHON_INSTALL_FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ezC3D.py
        ${CMAKE_CURRENT_BINARY_DIR}/_ezC3D.so)

# Configure setup.py and copy to output directory
 set(SETUP_PY_IN ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in)
 set(SETUP_PY_OUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
 configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})

# Declare install target for python
#install(TARGETS _ezC3D
#        COMMAND "${PYTHON_EXECUTABLE} setup.py"
#        COMPONENT swig-python)

# Install target to call setup.py
add_custom_target(install-python
        DEPENDS _ezC3D
COMMAND python ${SETUP_PY_OUT} install)
