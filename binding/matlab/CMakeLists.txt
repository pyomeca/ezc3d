cmake_minimum_required(VERSION 3.1)
find_package(Matlab COMPONENTS MX_LIBRARY REQUIRED)

# Get the files to compile
file(GLOB CPP_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp )

foreach(CPP_FILE ${CPP_FILES})
	get_filename_component(mex_lib_name ${CPP_FILE} NAME_WE)
	
	INCLUDE_DIRECTORIES(${Matlab_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../include ${CMAKE_CURRENT_SOURCE_DIR})
	add_library(${mex_lib_name} SHARED ${CPP_FILE})

	# Set the name of the library
	if(WIN32)
	  if (CMAKE_CL_64)
		  SET_TARGET_PROPERTIES(${mex_lib_name} PROPERTIES SUFFIX .mexw64)
	  else(CMAKE_CL_64)
		  SET_TARGET_PROPERTIES(${mex_lib_name} PROPERTIES SUFFIX .mexw32)
	  endif(CMAKE_CL_64)
	else(WIN32)
	  if (CMAKE_SIZEOF_VOID_P MATCHES "8")
		  SET_TARGET_PROPERTIES(${mex_lib_name} PROPERTIES SUFFIX .mexa64 PREFIX "")
	  else(CMAKE_SIZEOF_VOID_P MATCHES "8")
		  SET_TARGET_PROPERTIES(${mex_lib_name} PROPERTIES SUFFIX .mexglx PREFIX "")
	  endif (CMAKE_SIZEOF_VOID_P MATCHES "8")
	endif(WIN32)

	target_link_libraries(${mex_lib_name} ${PROJECT_NAME} ${Matlab_LIBRARIES})
	#install(TARGETS ${mex_lib_name} DESTINATION ${Matlab_ROOT_DIR}/toolbox/${mex_lib_name})
endforeach(CPP_FILE)












