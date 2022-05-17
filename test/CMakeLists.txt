project(${EZC3D_NAME}_test)

# Download gtest if necessary
set(RELATIVE_PATH_TO_GTEST external/gtest)
find_path(GTEST_INCLUDE_DIR "gtest.h" HINTS
    "${CMAKE_CURRENT_SOURCE_DIR}/../${RELATIVE_PATH_TO_GTEST}/googletest/include/gtest"
)

if((NOT GTEST_INCLUDE_DIR) OR (NOT EXISTS "${GTEST_INCLUDE_DIR}"))
    find_package(Git QUIET)
    if (NOT GIT_FOUND)
        message(FATAL_ERROR "Unable to find gtest and it could not be download since Git was not found!")
    endif()

    # we couldn't find the header files for gtest or they don't exist
    message("Unable to find gtest, I'll try to automatically download it")

    # we have a submodule setup for gtest, assume it is under external/gtest
    # now we need to clone this submodule
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init -- ${EZC3D_ROOT_FOLDER}/${RELATIVE_PATH_TO_GTEST}
                    WORKING_DIRECTORY ${EZC3D_ROOT_FOLDER}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
        message(FATAL_ERROR "${GIT_EXECUTABLE} submodule update --init -- ${EZC3D_ROOT_FOLDER}/external/gtest failed, please checkout submodules manually")
    endif()

    find_path(GTEST_INCLUDE_DIR "gtest.h" HINTS
        "${CMAKE_CURRENT_SOURCE_DIR}/../${RELATIVE_PATH_TO_GTEST}/googletest/include/gtest"
    )
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
set(BUILD_GMOCK OFF)
set(INSTALL_GTEST OFF)
add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../${RELATIVE_PATH_TO_GTEST}"
    "${CMAKE_BINARY_DIR}/gtest"
    EXCLUDE_FROM_ALL
)

##############
# Unit Tests
##############
enable_testing()

set(TEST_SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/test_ezc3d.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_math.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/test_modules.cpp"
)
add_executable(${PROJECT_NAME} "${TEST_SRC_FILES}")
target_link_libraries(${PROJECT_NAME} gtest_main ezc3d)

# This is so you can do 'make test' to see all your tests run, instead of
# manually running the executable ezc3d_test to see those specific tests.
add_test(UnitTests ${PROJECT_NAME})

if (CMAKE_BUILD_TYPE STREQUAL "Coverage")
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../.travis/cmake)

    include(CodeCoverage)
    setup_target_for_coverage(${EZC3D_NAME}_coverage
            "${PROJECT_NAME}" "coverage")

    SET(CMAKE_CXX_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
    SET(CMAKE_C_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
endif() #CMAKE_BUILD_TYPE STREQUAL "Coverage"

# Download and copy c3d test files from test suite of pyomeca
# Get the test suite from the pyomeca repository
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/ezc3d-testFiles-master)
    file(
        DOWNLOAD https://github.com/pyomeca/ezc3d-testFiles/archive/master.zip
        ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/ezc3d-testFiles-master.zip
        TIMEOUT 60 # seconds
        TLS_VERIFY ON
    )
endif()

# Get the test suite from c3d.org
set(TEST_FILES_TO_DOWNLOAD Sample01.zip Sample02.zip)
foreach(FILE ${TEST_FILES_TO_DOWNLOAD})
    # Get the test suite from c3d.org
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FILE})
        file(
            DOWNLOAD https://www.c3d.org/data/${FILE}
            ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FILE}
            TIMEOUT 60 # seconds
            TLS_VERIFY ON
        )
    endif()
endforeach()

# Copy all c3d to the test folder
set(TEST_FILES_UNZIP ezc3d-testFiles-master.zip Sample01.zip Sample02.zip)
foreach(FILE ${TEST_FILES_UNZIP})
    get_filename_component(FOLDER_NAME ${FILE} NAME_WE)
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FOLDER_NAME})
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FOLDER_NAME})
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar -xf ../${FILE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FOLDER_NAME}/
            )
    endif()
    file(GLOB C3D_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FOLDER_NAME}/*.c3d)
    if (NOT C3D_TEST_FILES)
        file(GLOB C3D_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/c3dFiles/${FOLDER_NAME}/${FOLDER_NAME}/*.c3d)
    endif()
    file(COPY ${C3D_TEST_FILES}
         DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/c3dTestFiles/)
endforeach()
