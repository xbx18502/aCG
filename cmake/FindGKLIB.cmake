# - Find GKLIB
# Find the GKLIB library
#
# This module defines:
#  GKLIB_FOUND        - True if GKLIB was found
#  GKLIB_INCLUDE_DIRS - Include directories for GKLIB
#  GKLIB_LIBRARIES    - Libraries to link against
#
# Variables used:
#  GKLIB_DIR          - Root directory of GKLIB installation

if(NOT GKLIB_INCLUDE_DIR)
  find_path(GKLIB_INCLUDE_DIR GKlib.h
    HINTS ${GKLIB_DIR} ENV GKLIB_DIR
    PATH_SUFFIXES include
    DOC "Directory where GKLIB header files are located"
  )
endif()

if(NOT GKLIB_LIBRARY)
  find_library(GKLIB_LIBRARY
    NAMES GKlib libGKlib gklib libgklib
    HINTS ${GKLIB_DIR} ENV GKLIB_DIR
    PATH_SUFFIXES lib64 lib
    DOC "GKLIB library location"
  )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GKLIB
  REQUIRED_VARS GKLIB_LIBRARY GKLIB_INCLUDE_DIR
)

if(GKLIB_FOUND)
  set(GKLIB_LIBRARIES ${GKLIB_LIBRARY})
  set(GKLIB_INCLUDE_DIRS ${GKLIB_INCLUDE_DIR})
endif()

mark_as_advanced(GKLIB_INCLUDE_DIR GKLIB_LIBRARY)
