# If Octave was installed from conda on Linux, the configuration process targets the wrong folder
if (LINUX)
    option(OCTAVE_FROM_CONDA "If Octave was installed from Conda (only relevant for Linux)" OFF)
else()
    option(OCTAVE_FROM_CONDA "If Octave was installed from Conda (only relevant for Linux)" ON)
endif()
if (NOT LINUX)
    MARK_AS_ADVANCED (
        OCTAVE_FROM_CONDA
    )
endif()

if(LINUX AND OCTAVE_FROM_CONDA)
# Searches for OCTAVE includes and library files
#
# Sets the variables
#   OCTAVE_FOUND
#   OCTAVE_INCLUDE_DIRS
#   OCTAVE_LIBRARIES

SET (OCTAVE_FOUND FALSE)

find_program( OCTAVE_CONFIG_EXECUTABLE
              NAMES octave-config
            )

if ( OCTAVE_CONFIG_EXECUTABLE )
    execute_process ( COMMAND ${OCTAVE_CONFIG_EXECUTABLE} -v
        OUTPUT_VARIABLE OCTAVE_VERSION_STRING
        OUTPUT_STRIP_TRAILING_WHITESPACE )

    FIND_PATH (OCTAVE_INCLUDE_DIRS oct.h
        /usr/include/octave-${OCTAVE_VERSION_STRING}/octave
        /usr/local/include/octave-${OCTAVE_VERSION_STRING}/octave
        $ENV{HOME}/local/include/octave-${OCTAVE_VERSION_STRING}/octave
        ${CMAKE_INSTALL_PREFIX}/include/octave-${OCTAVE_VERSION_STRING}/octave
        ${CMAKE_INSTALL_PREFIX}/Library/include/octave-${OCTAVE_VERSION_STRING}/octave
        $ENV{OCTAVE_INCLUDE_PATH}
    )

    IF (OCTAVE_INCLUDE_DIRS)
        SET (OCTAVE_FOUND TRUE)
    ENDIF (OCTAVE_INCLUDE_DIRS)

    IF (OCTAVE_FOUND)
       IF (NOT OCTAVE_FIND_QUIETLY)
          MESSAGE(STATUS "Found Octave: ${OCTAVE_INCLUDE_DIRS}")
       ENDIF (NOT OCTAVE_FIND_QUIETLY)
    ELSE (OCTAVE_FOUND)
       IF (OCTAVE_FIND_REQUIRED)
          MESSAGE(FATAL_ERROR "Could not find Octave")
       ENDIF (OCTAVE_FIND_REQUIRED)
    ENDIF (OCTAVE_FOUND)

    MARK_AS_ADVANCED (
        OCTAVE_FOUND
        OCTAVE_INCLUDE_DIRS
        OCTAVE_CONFIG_EXECUTABLE
    )

endif()

else(LINUX AND OCTAVE_FROM_CONDA)
# - Find Octave
# GNU Octave is a high-level interpreted language, primarily intended for numerical computations.
# available at http://www.gnu.org/software/octave/
#
# This module defines:
#  OCTAVE_INCLUDE_DIRS         - include path for mex.h, mexproto.h
#  OCTAVE_VERSION_STRING       - octave version string
#  OCTAVE_MAJOR_VERSION        - major version
#  OCTAVE_MINOR_VERSION        - minor version
#  OCTAVE_PATCH_VERSION        - patch version
#  OCTAVE_ROOT_DIR             - octave prefix

#=============================================================================
# Copyright 2013, Julien Schueller
# Copyright 2015, Martin Koehler
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project.
#=============================================================================

if(WIN32)
    SET(OCTAVE_ROOT_DIR "C:/Program Files/GNU Octave" CACHE PATH "Root directory of Octave where the file 'octave.vbs' can be found.")
	find_file(OCTAVE_VBS_FILE
        NAMES octave.vbs
        HINTS ${OCTAVE_ROOT_DIR}
    )
	
	mark_as_advanced(OCTAVE_VBS_FILE)
	if(NOT OCTAVE_VBS_FILE)
		message(FATAL_ERROR "OCTAVE_ROOT_DIR should point to root folder of Octave where the file 'octave.vbs' can be found.")
	endif()
endif()

find_program(OCTAVE_CONFIG_EXECUTABLE
    NAMES octave-config
	HINTS ${OCTAVE_ROOT_DIR}/mingw64/bin
)

if( OCTAVE_CONFIG_EXECUTABLE )
    execute_process( COMMAND ${OCTAVE_CONFIG_EXECUTABLE} -p OCTINCLUDEDIR
        OUTPUT_VARIABLE OCTAVE_INCLUDE_PATHS
        OUTPUT_STRIP_TRAILING_WHITESPACE )

    execute_process( COMMAND ${OCTAVE_CONFIG_EXECUTABLE} -p OCTLIBDIR
        OUTPUT_VARIABLE OCTAVE_LIBRARIES_PATHS
        OUTPUT_STRIP_TRAILING_WHITESPACE )

    execute_process( COMMAND ${OCTAVE_CONFIG_EXECUTABLE} -v
        OUTPUT_VARIABLE OCTAVE_VERSION_STRING
        OUTPUT_STRIP_TRAILING_WHITESPACE )

    if( OCTAVE_VERSION_STRING )
        string( REGEX REPLACE "([0-9]+)\\..*" "\\1" OCTAVE_MAJOR_VERSION ${OCTAVE_VERSION_STRING} )
        string( REGEX REPLACE "[0-9]+\\.([0-9]+).*" "\\1" OCTAVE_MINOR_VERSION ${OCTAVE_VERSION_STRING} )
        string( REGEX REPLACE "[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" OCTAVE_PATCH_VERSION ${OCTAVE_VERSION_STRING} )
    endif()
else( OCTAVE_CONFIG_EXECUTABLE )
    message(FATAL_ERROR "Octave-config could not be found. Please set the OCTAVE_CONFIG_EXECUTABLE variable to the 'octave-config' executable file")
endif()

find_path(OCTAVE_INCLUDE_DIRS
    NAMES oct.h
    HINTS ${OCTAVE_INCLUDE_PATHS}
)

if (WIN32)
    find_file(OCTAVE_LIBRARIES
        NAMES liboctinterp.dll.a
        HINTS ${OCTAVE_LIBRARIES_PATHS}
    )
elseif (APPLE)
    find_file(OCTAVE_LIBRARIES
        NAMES liboctinterp.dylib
        HINTS ${OCTAVE_LIBRARIES_PATHS}
    )
else()
    set(OCTAVE_LIBRARIES "")
endif()

mark_as_advanced(
    OCTAVE_ROOT_DIR
    OCTAVE_INCLUDE_DIRS
	OCTAVE_LIBRARIES 
    OCTAVE_VERSION_STRING
    OCTAVE_MAJOR_VERSION
    OCTAVE_MINOR_VERSION
    OCTAVE_PATCH_VERSION
	OCTAVE_CONFIG_EXECUTABLE
)
endif(LINUX AND OCTAVE_FROM_CONDA)
