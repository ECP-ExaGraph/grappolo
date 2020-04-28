# Nitin A. Gawande, PNNL
# Oct 19, 2018

PROJECT(GRAPPOLO C CXX)

option( DEFINE_RUNONGPU "Use of GPU" ON)
IF( DEFINE_RUNONGPU )
     MESSAGE( "GPU code enabled with CUDA" )
     FIND_PACKAGE(CUDA) 
     ADD_DEFINITIONS( -DRUNONGPU=1 )
ENDIF( DEFINE_RUNONGPU )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_minimum_required( VERSION 3.8 )

# Include cmake modules
list( APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" )

# Project options
option( BUILD_SHARED_LIBS     
        "Build shared libraries instead of static libraries."
        OFF )
option( OPTION_SELF_CONTAINED 
        "Create a self-contained install with all dependencies."
        OFF )
option( OPTION_BUILD_TESTS
        "Build tests."
        ON )
option( OPTION_BUILD_DOCS  
        "Build documentation."
        OFF )
option( OPTION_BUILD_EXAMPLES 
        "Build examples."
        OFF )

set( CMAKE_CXX_STANDARD 11 )

#set(CMAKE_C_COMPILER "gcc-8")
#set(CMAKE_CXX_COMPILER "g++-8")
#set(CMAKE_C_COMPILER "/share/apps/gcc/7.1/bin/gcc")
#set(CMAKE_CXX_COMPILER "/share/apps/gcc/7.1/bin/g++")

#set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -O3" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -fopenmp -O0" )

find_package(OpenMP)
if (OPENMP_FOUND)
    #set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# set varaibles using nomenclature used in Makefile
set( IOFOLDER InputsOutput )
set( COFOLDER BasicCommunitiesDetection )
set( UTFOLDER Utility )
set( CLFOLDER Coloring )
set( FSFOLDER FullSyncOptimization )
set( DEFINSTR DefineStructure )

IF( DEFINE_RUNONGPU )
     MESSAGE( "Use GPU code in gpu_graph directory" )
     set( GPUFOLDER gpu_graph ) 
     add_subdirectory( gpu_graph )
ENDIF( DEFINE_RUNONGPU )

set( IO_HEADERS
  ${CMAKE_SOURCE_DIR}/DefineStructure
)

add_subdirectory( InputsOutput )
add_subdirectory( BasicCommunitiesDetection )
add_subdirectory( Utility )
add_subdirectory( Coloring )
add_subdirectory( FullSyncOptimization )


#set( TARGET_1  convertFileToBinary )
#set( TARGET_2  driverForGraphClusteringApprox )
#set( TARGET_3  driverForColoring )
set( TARGET_4  driverForGraphClustering )
#set( TARGET_6  driverForGraphClusteringFastTrackResistance )
set( TARGET_7  driverForClusterComparison )
#set( TARGET_8  convertSNAPGroundTruthInformation )
set( TARGET_9  driverForMatrixReordering )
set( TARGET_10 driverForMatrixReorderingRcm )
#set( TARGET_11 driverForPartitioningWithMetis )
#set( TARGET_12 driverForMatrixReorderingND )

#add_executable( ${TARGET_1}  ${CMAKE_SOURCE_DIR}/${TARGET_1}.cpp )
#add_executable( ${TARGET_2}  ${CMAKE_SOURCE_DIR}/${TARGET_2}.cpp )
#add_executable( ${TARGET_3}  ${CMAKE_SOURCE_DIR}/${TARGET_3}.cpp )
add_executable( ${TARGET_4}  ${CMAKE_SOURCE_DIR}/${TARGET_4}.cpp )
#add_executable( ${TARGET_5}  ${CMAKE_SOURCE_DIR}/${TARGET_5}.cpp )
#add_executable( ${TARGET_6}  ${CMAKE_SOURCE_DIR}/${TARGET_6}.cpp )
add_executable( ${TARGET_7}  ${CMAKE_SOURCE_DIR}/${TARGET_7}.cpp )
#add_executable( ${TARGET_8}  ${CMAKE_SOURCE_DIR}/${TARGET_8}.cpp )
add_executable( ${TARGET_9}  ${CMAKE_SOURCE_DIR}/${TARGET_9}.cpp )
add_executable( ${TARGET_10} ${CMAKE_SOURCE_DIR}/${TARGET_10}.cpp )
#add_executable( ${TARGET_11} ${CMAKE_SOURCE_DIR}/${TARGET_11}.cpp )
#add_executable( ${TARGET_12} ${CMAKE_SOURCE_DIR}/${TARGET_12}.cpp )

target_include_directories( ${TARGET_4} 
    PUBLIC ${IO_HEADERS}
)

target_link_libraries( ${TARGET_4} 
    inout  
    basic_cd
    util
    full_syn_opt
    coloring
)

target_link_libraries( ${TARGET_7} 
    inout
    util 
)

target_link_libraries( ${TARGET_9} 
    inout  
    basic_cd
    util
    full_syn_opt
    coloring
)

target_link_libraries( ${TARGET_10} 
    inout  
    basic_cd
    util
    full_syn_opt
    coloring
)

#target_link_libraries( ${TARGET_10} LINK_PUBLIC
#     ${inout} ${basic_cd} ${util} ${full_syn_opt} ${full_syn_opt} ${coloring} )



install( TARGETS
  ${TARGET_4}
  ${TARGET_7}
  ${TARGET_9}
  ${TARGET_10}
  DESTINATION bin
)

#target_include_directories( ${TARGET_4}  PUBLIC ${CMAKE_SOURCE_DIR}/${DEFINSTR} )
#target_include_directories( ${TARGET_7}  PUBLIC ${CMAKE_SOURCE_DIR}/${DEFINSTR} )
#target_include_directories( ${TARGET_9}  PUBLIC ${CMAKE_SOURCE_DIR}/${DEFINSTR} )
#target_include_directories( ${TARGET_10} PUBLIC ${CMAKE_SOURCE_DIR}/${DEFINSTR} )

