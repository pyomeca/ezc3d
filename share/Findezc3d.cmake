# - Find ezc3d
# Find the native ezc3d includes and libraries
#
#  ezc3d_INCLUDE_DIR - where to find ezc3d.h, etc.
#  ezc3d_LIBRARIES   - List of libraries when using ezc3d.
#  ezc3d_FOUND       - True if ezc3d is found.

if (ezc3d_INCLUDE_DIR)
  # Already in cache, be silent
  set (ezc3d_FIND_QUIETLY TRUE)
endif (ezc3d_INCLUDE_DIR)

find_path (ezc3d_INCLUDE_DIR "ezc3d.h" PATHS ${CMAKE_INSTALL_PREFIX}/include/ezc3d)
find_library (ezc3d_LIBRARY NAMES ezc3d ezc3d_debug PATHS ${CMAKE_INSTALL_PREFIX}/lib/ezc3d)


# handle the QUIETLY and REQUIRED arguments and set ezc3d_FOUND to TRUE if 
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (ezc3d DEFAULT_MSG 
  ezc3d_LIBRARY
  ezc3d_INCLUDE_DIR
)

