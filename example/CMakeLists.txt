set(MASTER_PROJECT_NAME ${PROJECT_NAME})

project(${MASTER_PROJECT_NAME}_example)
add_executable(${PROJECT_NAME} "main.cpp")
add_dependencies(${PROJECT_NAME} ${MASTER_PROJECT_NAME})

# Headers
target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

# Libraries linker
get_target_property(DEBUG_POSTFIX ${MASTER_PROJECT_NAME} DEBUG_POSTFIX)
IF(CMAKE_BUILD_TYPE MATCHES "Debug")
    SET(${MASTER_PROJECT_NAME}_LIB_NAME ${MASTER_PROJECT_NAME}${DEBUG_POSTFIX})
else()
    SET(${MASTER_PROJECT_NAME}_LIB_NAME ${MASTER_PROJECT_NAME})
endif()
if (APPLE)
    string(CONCAT ${MASTER_PROJECT_NAME}_LIB_NAME "lib" ${${MASTER_PROJECT_NAME}_LIB_NAME} ".dylib")
elseif(UNIX)
    string(CONCAT ${MASTER_PROJECT_NAME}_LIB_NAME "lib" ${${MASTER_PROJECT_NAME}_LIB_NAME} ".so")
elseif(WIN32)
    message("Compilation won't work on Windows, to be corrected later...")
    string(CONCAT ${MASTER_PROJECT_NAME}_LIB_NAME ${${MASTER_PROJECT_NAME}_LIB_NAME} ".dll")
else()
    message(FATAL_ERROR "Unrecognized system")
endif()
set(${MASTER_PROJECT_NAME}_LIBRARY ${CMAKE_BINARY_DIR}/${${MASTER_PROJECT_NAME}_LIB_NAME})
target_link_libraries(${PROJECT_NAME} ${${MASTER_PROJECT_NAME}_LIBRARY})

# Copy the c3d of the example
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/markers_analogs.c3d
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
