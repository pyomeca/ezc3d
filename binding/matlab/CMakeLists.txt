project(${EZC3D_NAME}_matlab)
cmake_minimum_required(VERSION 3.8)

find_package(Matlab REQUIRED COMPONENTS MX_LIBRARY)

# Installation path
set(Matlab_${EZC3D_NAME}_INSTALL_DIR "${Matlab_ROOT_DIR}/toolbox" CACHE PATH
    "Toolbox path to install, please note that the default directory may
    require administrator access"
)

# Get the file to compile
set(CPP_FILE_Read ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ezc3dRead.cpp
)
set(CPP_FILE_Write ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ezc3dWrite.cpp
)

foreach(SUFFIXES Read Write)
    # Add project
    add_library(${PROJECT_NAME}_${SUFFIXES} SHARED
        "${CPP_FILE_${SUFFIXES}}"
        "${CMAKE_CURRENT_SOURCE_DIR}/Matlabdef.def"
    )
    # Reset the name of the output
    if(WIN32)
        SET_TARGET_PROPERTIES(${PROJECT_NAME}_${SUFFIXES} PROPERTIES
            OUTPUT_NAME "${EZC3D_NAME}${SUFFIXES}"
        )
    else(WIN32)
        SET_TARGET_PROPERTIES(${PROJECT_NAME}_${SUFFIXES} PROPERTIES
            LIBRARY_OUTPUT_NAME "${EZC3D_NAME}${SUFFIXES}"
        )
    endif(WIN32)

    # Add headers
    target_include_directories(${PROJECT_NAME}_${SUFFIXES} PRIVATE
        "${Matlab_INCLUDE_DIRS}"
    )

    # Set the name of the library
    if(WIN32)
        if (CMAKE_CL_64)
            set_target_properties(${PROJECT_NAME}_${SUFFIXES} PROPERTIES
                SUFFIX ".mexw64"
            )
        else(CMAKE_CL_64)
            set_target_properties(${PROJECT_NAME}_${SUFFIXES} PROPERTIES
                SUFFIX ".mexw32"
            )
        endif(CMAKE_CL_64)
    elseif(APPLE)
        if (CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
            set_target_properties(${PROJECT_NAME}_${SUFFIXES} PROPERTIES
                SUFFIX ".mexmaci64"
                PREFIX ""
            )
        elseif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "arm64") 
            set_target_properties(${PROJECT_NAME}_${SUFFIXES} PROPERTIES
                SUFFIX ".mexmaca64"
                PREFIX ""
            )
        else()
            message(FATAL_ERROR "Unknown architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}") 
        endif ()
    else()
        if (CMAKE_SIZEOF_VOID_P MATCHES "8")
            set_target_properties(${PROJECT_NAME}_${SUFFIXES} PROPERTIES
                SUFFIX ".mexa64"
                PREFIX ""
            )
        else(CMAKE_SIZEOF_VOID_P MATCHES "8")
            set_target_properties(${PROJECT_NAME}_${SUFFIXES} PROPERTIES
                SUFFIX ".mexglx"
                PREFIX ""
            )
        endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
    endif()


    # Set runtime path for linux
    set_target_properties(${PROJECT_NAME}_${SUFFIXES} PROPERTIES
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )

    # Linker and instalation
    target_link_libraries(${PROJECT_NAME}_${SUFFIXES}
        "${EZC3D_NAME}"
        "${Matlab_MX_LIBRARY}"
        "${Matlab_MEX_LIBRARY}"
    )

    # install target
    install(TARGETS ${PROJECT_NAME}_${SUFFIXES}
        DESTINATION "${Matlab_${EZC3D_NAME}_INSTALL_DIR}/${PROJECT_NAME}")
endforeach(SUFFIXES)

if(WIN32)
    # install ezc3d library
    install(TARGETS ${EZC3D_NAME}
        ARCHIVE DESTINATION ${Matlab_${EZC3D_NAME}_INSTALL_DIR}/${PROJECT_NAME}
        RUNTIME DESTINATION ${Matlab_${EZC3D_NAME}_INSTALL_DIR}/${PROJECT_NAME}
        LIBRARY DESTINATION ${Matlab_${EZC3D_NAME}_INSTALL_DIR}/${PROJECT_NAME}
    )
endif()

# install utils
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/ezc3dNewParam.m"
    DESTINATION "${Matlab_${EZC3D_NAME}_INSTALL_DIR}/${PROJECT_NAME}")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/ezc3dAddEvent.m"
    DESTINATION "${Matlab_${EZC3D_NAME}_INSTALL_DIR}/${PROJECT_NAME}")
