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
    )

endif()
