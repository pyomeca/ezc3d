# Include python
find_package(PYTHON3 REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
find_package(Numpy)
include_directories(${NUMPY_INCLUDES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)
if(WIN32)
	link_directories(${Python3_RUNTIME_LIBRARY_DIRS}/libs)
endif()

# Add swig module
set(CMAKE_SWIG_FLAGS "")
if(NUMPY_FOUND)
	set(I_SOURCE_FILE ${PROJECT_NAME}_python.i)
else()
	message(WARNING "Numpy has not been found, I will compile without the easy accessors interface")
	set(I_SOURCE_FILE ../${PROJECT_NAME}.i)
endif()
set_property(SOURCE ${I_SOURCE_FILE} PROPERTY CPLUSPLUS ON)
SWIG_ADD_LIBRARY(${PROJECT_NAME} TYPE MODULE LANGUAGE python SOURCES ${I_SOURCE_FILE})
swig_link_libraries(${PROJECT_NAME} ${PROJECT_NAME} ${PYTHON_LIBRARIES} ${NUMPY_LIBRARIES})

SET_TARGET_PROPERTIES(_${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${${PROJECT_NAME}_BIN_FOLDER})
SET_TARGET_PROPERTIES(_${PROJECT_NAME} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# Install to the proper directories
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.py 
        DESTINATION ${Python3_SITELIB}/${PROJECT_NAME}
        RENAME __init__.py)

# For some reason, I could not install the library directly into subfolder
# Some day this should be fixed...
INSTALL(TARGETS _${PROJECT_NAME}
		DESTINATION ${Python3_SITELIB})  

