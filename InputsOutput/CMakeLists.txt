# grappolo 

set( IO_HEADERS
  ${CMAKE_SOURCE_DIR}/DefineStructure
)

set( IOFOLDER_SRC
  ${CMAKE_SOURCE_DIR}/${IOFOLDER}/loadBinary.cpp
  ${CMAKE_SOURCE_DIR}/${IOFOLDER}/loadDimacs.cpp
  ${CMAKE_SOURCE_DIR}/${IOFOLDER}/loadEdgeList.cpp
  ${CMAKE_SOURCE_DIR}/${IOFOLDER}/loadMatrixMarket.cpp
  ${CMAKE_SOURCE_DIR}/${IOFOLDER}/loadMetis.cpp
  ${CMAKE_SOURCE_DIR}/${IOFOLDER}/loadPajekFormat.cpp
  ${CMAKE_SOURCE_DIR}/${IOFOLDER}/loadSNAP.cpp
  ${CMAKE_SOURCE_DIR}/${IOFOLDER}/parseInputParameters.cpp
  ${CMAKE_SOURCE_DIR}/${IOFOLDER}/writeBinary.cpp
  ${CMAKE_SOURCE_DIR}/${IOFOLDER}/writeMatrixMarket.cpp
  ${CMAKE_SOURCE_DIR}/${IOFOLDER}/writeSimple.cpp
)

add_library( inout STATIC
  ${IOFOLDER_SRC} 
)

target_include_directories(inout PUBLIC ${IO_HEADERS} )

target_link_libraries( inout )

install( TARGETS
  inout
  DESTINATION lib
)

install( FILES
  ${IO_HEADERS}/input_output.h
  DESTINATION include
)

