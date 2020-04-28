# grappolo

set( GPU_HEADERS
  ${CMAKE_SOURCE_DIR}/gpu_graph
)

set( GPUFOLDER_SRC
     #${CMAKE_SOURCE_DIR}/${GPUFOLDER}/gpu_main.cpp
     ${CMAKE_SOURCE_DIR}/${GPUFOLDER}/graphHOST.cpp

     ${CMAKE_SOURCE_DIR}/${GPUFOLDER}/aggregateCommunity.cu
     ${CMAKE_SOURCE_DIR}/${GPUFOLDER}/assignGraph.cu
     ${CMAKE_SOURCE_DIR}/${GPUFOLDER}/binWiseGaussSeidel.cu
     ${CMAKE_SOURCE_DIR}/${GPUFOLDER}/communityGPU.cu
     ${CMAKE_SOURCE_DIR}/${GPUFOLDER}/computeModularity.cu
     ${CMAKE_SOURCE_DIR}/${GPUFOLDER}/computeTime.cu
     ${CMAKE_SOURCE_DIR}/${GPUFOLDER}/coreutility.cu
     ${CMAKE_SOURCE_DIR}/${GPUFOLDER}/gatherInformation.cu
     ${CMAKE_SOURCE_DIR}/${GPUFOLDER}/graphGPU.cu
     ${CMAKE_SOURCE_DIR}/${GPUFOLDER}/independentKernels.cu
     ${CMAKE_SOURCE_DIR}/${GPUFOLDER}/preprocessing.cu
   )

#CUDA_ADD_LIBRARY( gpu_graph 
#  SHARED
#  ${GPUFOLDER_SRC}
#  OPTIONS -arch sm_35
#)

CUDA_ADD_EXECUTABLE( gpu_graph 
  gpu_main.cpp
  ${GPUFOLDER_SRC}
  #OPTIONS -arch sm_35
  OPTIONS -arch sm_37
  DEBUG -g -G
)

target_include_directories( gpu_graph 
     PUBLIC 
     ${GPU_HEADERS}
     ${IO_HEADERS}
)

target_link_libraries( gpu_graph 
     inout
     util
     ${CUDA_LIBRARIES}
)

install( TARGETS
  gpu_graph
  #DESTINATION lib
  DESTINATION bin
)

install( FILES
  ${GPU_HEADERS}/commonconstants.h
  ${GPU_HEADERS}/communityGPU.h
  ${GPU_HEADERS}/devconstants.h
  ${GPU_HEADERS}/graphGPU.h
  ${GPU_HEADERS}/graphHOST.h
  ${GPU_HEADERS}/hashitem.h
  ${GPU_HEADERS}/hostconstants.h
  ${GPU_HEADERS}/myutility.h
  ${GPU_HEADERS}/openaddressing.h
  DESTINATION include
)

