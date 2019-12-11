set(MASTER_PROJECT_NAME ${PROJECT_NAME})

project(${MASTER_PROJECT_NAME}_example)
add_executable(${PROJECT_NAME} "main.cpp")
add_dependencies(${PROJECT_NAME} ${MASTER_PROJECT_NAME})

# Headers
target_include_directories(
    ${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${EZC3D_BINARY_DIR}/include
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
set(${MASTER_PROJECT_NAME}_LIBRARY ${EZC3D_BINARY_DIR}/${${MASTER_PROJECT_NAME}_LIB_NAME})
target_link_libraries(${PROJECT_NAME} ${${MASTER_PROJECT_NAME}_LIBRARY})

# Get the test suite from the pyomeca repository
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/ezc3d-testFiles-master)
    file(
        DOWNLOAD https://github.com/pyomeca/ezc3d-testFiles/archive/master.zip
        ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/ezc3d-testFiles-master.zip
        TIMEOUT 60 # seconds
        TLS_VERIFY ON
    )
endif()

# Copy all c3d to the test folder
set(FILE ezc3d-testFiles-master.zip)
get_filename_component(FOLDER_NAME ${FILE} NAME_WE)
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FOLDER_NAME})
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FOLDER_NAME})
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar -xf ../${FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FOLDER_NAME}/
        )
endif()
file(GLOB C3D_EXAMPLE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FOLDER_NAME}/*.c3d)
if (NOT C3D_EXAMPLE_FILES)
    file(GLOB C3D_EXAMPLE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FOLDER_NAME}/${FOLDER_NAME}/*.c3d)
endif()
file(COPY ${C3D_EXAMPLE_FILES}
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/c3dExampleFiles/)
