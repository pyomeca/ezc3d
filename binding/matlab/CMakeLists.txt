set(MASTER_PROJECT_NAME ${PROJECT_NAME})

project(${MASTER_PROJECT_NAME}_matlab)
cmake_minimum_required(VERSION 3.1)

find_package(Matlab REQUIRED COMPONENTS MX_LIBRARY)

# Get the file to compile
set(CPP_FILE_Read ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
             ${CMAKE_CURRENT_SOURCE_DIR}/ezc3dRead.cpp)
 set(CPP_FILE_Write ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
              ${CMAKE_CURRENT_SOURCE_DIR}/ezc3dWrite.cpp)

foreach(SUFFIXES Read Write)
    # Add project
    add_library(${PROJECT_NAME}_${SUFFIXES} SHARED ${CPP_FILE_${SUFFIXES}} ${CMAKE_CURRENT_SOURCE_DIR}/Matlabdef.def)

    target_link_libraries(${PROJECT_NAME}_${SUFFIXES} ezc3d)

    # Reset the name of the output
	if(WIN32)
		SET_TARGET_PROPERTIES(${PROJECT_NAME}_${SUFFIXES} PROPERTIES OUTPUT_NAME ${MASTER_PROJECT_NAME}${SUFFIXES})
	else(WIN32)
		SET_TARGET_PROPERTIES(${PROJECT_NAME}_${SUFFIXES} PROPERTIES LIBRARY_OUTPUT_NAME ${MASTER_PROJECT_NAME}${SUFFIXES})
	endif(WIN32)

    # Add headers
    target_include_directories(${PROJECT_NAME}_${SUFFIXES} PUBLIC
        ${Matlab_INCLUDE_DIRS}
        ${EZC3D_BINARY_DIR}/include
    )

    # Set the name of the library
    if(WIN32)
      if (CMAKE_CL_64)
          SET_TARGET_PROPERTIES(${PROJECT_NAME}_${SUFFIXES} PROPERTIES SUFFIX .mexw64)
      else(CMAKE_CL_64)
          SET_TARGET_PROPERTIES(${PROJECT_NAME}_${SUFFIXES} PROPERTIES SUFFIX .mexw32)
      endif(CMAKE_CL_64)
    elseif(APPLE)
      if (CMAKE_SIZEOF_VOID_P MATCHES "8")
          SET_TARGET_PROPERTIES(${PROJECT_NAME}_${SUFFIXES} PROPERTIES SUFFIX .mexmaci64 PREFIX "")
      else(CMAKE_SIZEOF_VOID_P MATCHES "8")
          SET_TARGET_PROPERTIES(${PROJECT_NAME}_${SUFFIXES} PROPERTIES SUFFIX .mexmaci64 PREFIX "")
      endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
    else()
      if (CMAKE_SIZEOF_VOID_P MATCHES "8")
          SET_TARGET_PROPERTIES(${PROJECT_NAME}_${SUFFIXES} PROPERTIES SUFFIX .mexa64 PREFIX "")
      else(CMAKE_SIZEOF_VOID_P MATCHES "8")
          SET_TARGET_PROPERTIES(${PROJECT_NAME}_${SUFFIXES} PROPERTIES SUFFIX .mexglx PREFIX "")
      endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
    endif(WIN32)

    # Set runtime path for linux
    SET_TARGET_PROPERTIES(${PROJECT_NAME}_${SUFFIXES} PROPERTIES INSTALL_RPATH ${${MASTER_PROJECT_NAME}_BIN_FOLDER})
    SET_TARGET_PROPERTIES(${PROJECT_NAME}_${SUFFIXES} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

    # Linker and instalation
    target_link_libraries(${PROJECT_NAME}_${SUFFIXES} ${MASTER_PROJECT_NAME} ${Matlab_MX_LIBRARY} ${Matlab_MEX_LIBRARY})
	if(WIN32)
		install(TARGETS ${PROJECT_NAME}_${SUFFIXES}
			RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${${MASTER_PROJECT_NAME}_BIN_FOLDER}
		)
	else(WIN32)
                set(Matlab_${MASTER_PROJECT_NAME}_INSTALL_DIR ${Matlab_ROOT_DIR}/toolbox CACHE PATH "Toolbox path to install, please note that the default directory may require administrator access")
		install(TARGETS ${PROJECT_NAME}_${SUFFIXES}
			DESTINATION ${Matlab_${MASTER_PROJECT_NAME}_INSTALL_DIR}/${PROJECT_NAME}
		)
	endif(WIN32)
	
endforeach(SUFFIXES)





