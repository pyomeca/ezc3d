set(MASTER_PROJECT_NAME ${PROJECT_NAME})

project(${MASTER_PROJECT_NAME}_example)
add_executable(${PROJECT_NAME} "main.cpp")

# Headers
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

# ezc3d Libraries linker
get_target_property(DEBUG_POSTFIX ${MASTER_PROJECT_NAME} DEBUG_POSTFIX)
IF(CMAKE_BUILD_TYPE MATCHES "Debug")
    SET(EZC3D_LIB_NAME ${MASTER_PROJECT_NAME}${DEBUG_POSTFIX})
else()
    SET(EZC3D_LIB_NAME ${MASTER_PROJECT_NAME})
endif()
if (UNIX)
    string(CONCAT EZC3D_LIB_NAME "lib" ${EZC3D_LIB_NAME} ".so")
elseif(_WIN32)
    message("TO BE TESTED")
    string(CONCAT EZC3D_LIB_NAME ${EZC3D_LIB_NAME} ".dll")
endif()
set(EZC3D_LIBRARY ${CMAKE_BINARY_DIR}/${EZC3D_LIB_NAME})
target_link_libraries(${PROJECT_NAME} ${EZC3D_LIBRARY})

# Copy the c3d of the example
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/markers_analogs.c3d
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
