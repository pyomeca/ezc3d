set(EXAMPLE_FILES
    "create_and_read_example.cpp"
    "modify_c3d_example.cpp"
    "force_plate_example.cpp"
)

foreach(FILE ${EXAMPLE_FILES})
    # Get the name of the current file
    get_filename_component(FILE_NAME ${FILE} NAME_WE)

    add_executable(${FILE_NAME} ${FILE})
    target_link_libraries(${FILE_NAME} ezc3d)
endforeach()

# Get the test suite from the pyomeca repository
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/ezc3d-testFiles-master)
    file(
        DOWNLOAD https://github.com/pyomeca/ezc3d-testFiles/archive/master.zip
        ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/ezc3d-testFiles-master.zip
        TIMEOUT 60 # seconds
        TLS_VERIFY ON
    )
endif()

# Copy all c3d to the example folder
set(FILE ezc3d-testFiles-master.zip)
get_filename_component(FOLDER_NAME ${FILE} NAME_WE)
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FOLDER_NAME})
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FOLDER_NAME})
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar -xf ../${FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FOLDER_NAME}/
    )
endif()
file(GLOB C3D_EXAMPLE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FOLDER_NAME}/*.c3d
)
if (NOT C3D_EXAMPLE_FILES)
    file(GLOB C3D_EXAMPLE_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FOLDER_NAME}/${FOLDER_NAME}/*.c3d
    )
endif()
file(COPY ${C3D_EXAMPLE_FILES}
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/c3dExampleFiles/
)
