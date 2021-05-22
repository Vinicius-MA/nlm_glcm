cmake_minimum_required(VERSION 2.6)

# SETUP A MAKE CLEAN TO CLEAN EVERYTHING
SET_DIRECTORY_PROPERTIES( PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES 
   "CMakeCache.txt;cmake_install.cmake" )

# DEFAULT BUILD: RELEASE
if (NOT CMAKE_BUILD_TYPE)
   message(STATUS "No build type selected, default to Release")
   set(CMAKE_BUILD_TYPE "Release")
endif()

# ADD LOCAL PATH TO SEARCH MODULES
LIST (APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_PREFIX}")
SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

# PNG
FIND_PACKAGE(PNG REQUIRED)
IF(PNG_FOUND)
   INCLUDE_DIRECTORIES(${PNG_INCLUDE_DIR})
   #  LINK_DIRECTORIES(${PNG_LIBRARY})
   SET( LIBS ${LIBS} ${PNG_LIBRARIES})
ENDIF()

# FFTW3 
FIND_PACKAGE(FFTW3 COMPONENTS single REQUIRED)
IF(FFTW3_FOUND)
   INCLUDE_DIRECTORIES(${FFTW3_INCLUDE_DIR})
   SET( LIBS ${LIBS} ${FFTW3_LIBRARIES})
ENDIF()

# MATH
IF (UNIX)
	SET( LIBS "${LIBS}" m )
ENDIF()

# OpenMP 
FIND_PACKAGE( OpenMP )
IF(OPENMP_FOUND)
   MESSAGE("OPENMP FOUND")
   SET( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}" )
ENDIF()

# FLAGS
if(CMAKE_COMPILER_IS_GNUCC)
      SET( EXTRA_COMPILER_FLAGS "${EXTRA_COMPILER_FLAGS} -ftree-vectorize -funroll-loops -ansi -fPIC" )
endif()


######################

SET( COMMON_SRC 
         Utilities/LibImages.cpp 
         Utilities/Utilities.cpp 
         Utilities/mt19937ar.c 
         Utilities/LibImages_io.cpp 
         Utilities/io_png.c 
    )



# NLDD BINARY
add_executable (NLDD
                main_NLDD.cpp DDID/RunDDID.cpp NlBayes/NlBayes.cpp NlBayes/LibMatrix.cpp
                ${COMMON_SRC}
               )
target_link_libraries(NLDD ${LIBS})
SET_TARGET_PROPERTIES(NLDD PROPERTIES
      COMPILE_FLAGS "${OpenMP_CXX_FLAGS} ${EXTRA_COMPILER_FLAGS}")


# DDID BINARY
add_executable (DDID
                main_DDID.cpp DDID/RunDDID.cpp NlBayes/NlBayes.cpp NlBayes/LibMatrix.cpp 
                ${COMMON_SRC}
                )
target_link_libraries(DDID ${LIBS})
SET_TARGET_PROPERTIES(DDID PROPERTIES
   COMPILE_FLAGS "${OpenMP_CXX_FLAGS} ${EXTRA_COMPILER_FLAGS}")



# NLBAYES BINARY
add_executable (NLBAYES
                main_NLBayes.cpp NlBayes/NlBayes.cpp NlBayes/LibMatrix.cpp
                ${COMMON_SRC}
                )
target_link_libraries(NLBAYES ${LIBS})
SET_TARGET_PROPERTIES(NLBAYES PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS} ${EXTRA_COMPILER_FLAGS}")


