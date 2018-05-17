# Include python
find_package(PYTHON3 REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})
find_package(Numpy REQUIRED)
include_directories(${NUMPY_INCLUDES})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# Add swig module
#set(CMAKE_SWIG_FLAGS "")
set_property(SOURCE ${PROJECT_NAME}_python.i PROPERTY CPLUSPLUS ON)
SWIG_ADD_LIBRARY(${PROJECT_NAME} TYPE MODULE LANGUAGE python SOURCES ${PROJECT_NAME}_python.i)
swig_link_libraries(${PROJECT_NAME} ${PROJECT_NAME} ${PYTHON_LIBRARIES} ${NUMPY_LIBRARIES})

# Install to the proper directories
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.py 
        DESTINATION ${Python3_SITELIB}/${PROJECT_NAME}
        RENAME __init__.py)

# For some reason, I could not install the library directly into subfolder
# Some day this should be fixed...
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_${PROJECT_NAME}.so
        DESTINATION ${Python3_SITELIB})

