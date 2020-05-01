#GCC Compilers:
CC  = gcc-9
CPP = g++-9
CFLAGS   = -g -Ofast -fopenmp -std=c99 
CPPFLAGS = -g -Ofast -fopenmp -std=c++11
#-Ofast

METIS_HOME = $(HOME)/metis-5.1.0
METIS_INCLUDE = -I$(METIS_HOME)/include
METIS_LIB = -L$(METIS_HOME)/lib -lmetis -lm

LDFLAGS  = $(CPPFLAGS)
#INCLUDES = ./DefineStructure/
INCLUDES = ./DefineStructure/ $(METIS_INCLUDE)
IOFOLDER = ./InputsOutput
COFOLDER = ./BasicCommunitiesDetection
UTFOLDER = ./Utility
CLFOLDER = ./Coloring
FSFOLDER = ./FullSyncOptimization
LIBS     = -lm

TARGET_1 = convertFileToBinary
TARGET_2 = driverForGraphClusteringApprox
TARGET_3 = driverForColoring
TARGET_4 = driverForGraphClustering
TARGET_6 = driverForGraphClusteringFastTrackResistance
TARGET_7 = driverForClusterComparison
TARGET_8 = convertSNAPGroundTruthInformation
TARGET_9 = driverForMatrixReordering
TARGET_10 = driverForMatrixReorderingRcm
TARGET_11 = driverForPartitioningWithMetis
TARGET_12 = driverForMatrixReorderingND

#TARGET   = $(TARGET_1) $(TARGET_2) $(TARGET_4) $(TARGET_7) $(TARGET_8) $(TARGET_9) $(TARGET_10)
#TARGET   = $(TARGET_4) $(TARGET_7) $(TARGET_9) $(TARGET_10) $(TARGET_11) $(TARGET_12) 
#TARGET   = $(TARGET_4) $(TARGET_7) 
TARGET   = $(TARGET_4) $(TARGET_6) $(TARGET_7)

# $(TARGET_3) $(TARGET_4) $(TARGET_5)
#TARGET = $(TARGET_1) $(TARGET_2) $(TARGET_3) $(TARGET_4)

TARGET_H5 = driverForGraphClusteringH5

#IOBJECTS = loadBinary.o loadMetis.o loadMatrixMarket.o \
          loadEdgeList.o 
IOFILES = $(wildcard $(IOFOLDER)/*.cpp)
IOOBJECTS = $(addprefix $(IOFOLDER)/,$(notdir $(IOFILES:.cpp=.o)))

COFILES = $(wildcard $(COFOLDER)/*.cpp)
COOBJECTS = $(addprefix $(COFOLDER)/,$(notdir $(COFILES:.cpp=.o)))

UTFILES = $(wildcard $(UTFOLDER)/*.cpp)
UTOBJECTS = $(addprefix $(UTFOLDER)/,$(notdir $(UTFILES:.cpp=.o)))

CLFILES = $(wildcard $(CLFOLDER)/*.cpp)
CLOBJECTS = $(addprefix $(CLFOLDER)/,$(notdir $(CLFILES:.cpp=.o)))

CLFILES2 = coloringDistanceOne.o equitableColoringDistanceOne.o coloringUtils.cpp
CLOBJECTS2 = $(addprefix $(CLFOLDER)/,$(notdir $(CLFILES2:.cpp=.o)))
 
FSFILES = $(wildcard $(FSFOLDER)/*.cpp)
FSOBJECTS = $(addprefix $(FSFOLDER)/,$(notdir $(FSFILES:.cpp=.o)))



#OBJECTS  = RngStream.o utilityFunctions.o parseInputFiles.o \
           writeGraphDimacsFormat.o buildNextPhase.o \
           coloringDistanceOne.o utilityClusteringFunctions.o equitableColoringDistanceOne.o\
           parallelLouvainMethod.o parallelLouvainMethodNoMap.o \
           parallelLouvainMethodScale.o parallelLouvainWithColoring.o parallelLouvainWithColoringNoMap.o \
	   louvainMultiPhaseRun.o parseInputParameters.o vertexFollowing.o parallelLouvianMethodApprox.o runMultiPhaseBasicApprox.o

all: $(TARGET) 
coloring: $(TARGET_3)
hdf5: $(TARGET_H5)

mtx: $(TARGET_9) $(TARGET_10)


#message

$(TARGET_1): $(IOOBJECTS) $(UTOBJECTS) $(TARGET_1).o
	$(CPP) $(LDFLAGS) -o ./bin/$(TARGET_1) $(UTOBJECTS) $(IOOBJECTS) $(TARGET_1).o

$(TARGET_3): $(IOOBJECTS) $(CLOBJECTS2) $(UTOBJECTS) $(TARGET_3).o
	$(CPP) $(LDFLAGS) -o ./bin/$(TARGET_3) $(IOOBJECTS) $(UTOBJECTS) $(CLOBJECTS2) $(TARGET_3).o

$(TARGET_2): $(IOOBJECTS) $(COOBJECTS) $(UTOBJECTS) $(FSOBJECTS) $(CLOBJECTS) $(TARGET_2).o
	$(CPP) $(LDFLAGS) -o ./bin/$(TARGET_2) $(TARGET_2).o $(FSOBJECTS) $(IOOBJECTS) $(COOBJECTS) $(UTOBJECTS) $(CLOBJECTS) $(LIBS)

$(TARGET_4): $(IOOBJECTS) $(COOBJECTS) $(UTOBJECTS) $(FSOBJECTS) $(CLOBJECTS) $(TARGET_4).o
	$(CPP) $(LDFLAGS) -o ./bin/$(TARGET_4) $(TARGET_4).o $(FSOBJECTS) $(IOOBJECTS) $(COOBJECTS) $(UTOBJECTS) $(CLOBJECTS) $(LIBS)

$(TARGET_6): $(IOOBJECTS) $(COOBJECTS) $(UTOBJECTS) $(FSOBJECTS) $(CLOBJECTS) $(TARGET_6).o
	$(CPP) $(LDFLAGS) -o ./bin/$(TARGET_6) $(TARGET_6).o $(FSOBJECTS) $(IOOBJECTS) $(COOBJECTS) $(UTOBJECTS) $(CLOBJECTS) $(LIBS)

$(TARGET_7): $(UTOBJECTS) $(TARGET_7).o
	$(CPP) $(LDFLAGS) -o ./bin/$(TARGET_7) $(UTOBJECTS) $(IOOBJECTS) $(TARGET_7).o

$(TARGET_8): $(UTOBJECTS) $(TARGET_8).o
	$(CPP) $(LDFLAGS) -o ./bin/$(TARGET_8) $(UTOBJECTS) $(IOOBJECTS) $(TARGET_8).o

$(TARGET_9): $(IOOBJECTS) $(COOBJECTS) $(UTOBJECTS) $(FSOBJECTS) $(CLOBJECTS) $(TARGET_9).o
	$(CPP) $(LDFLAGS) -o ./bin/$(TARGET_9) $(TARGET_9).o $(FSOBJECTS) $(IOOBJECTS) $(COOBJECTS) $(UTOBJECTS) $(CLOBJECTS) $(LIBS)

$(TARGET_10): $(IOOBJECTS) $(COOBJECTS) $(UTOBJECTS) $(FSOBJECTS) $(CLOBJECTS) $(TARGET_10).o
	$(CPP) $(LDFLAGS) -o ./bin/$(TARGET_10) $(TARGET_10).o $(FSOBJECTS) $(IOOBJECTS) $(COOBJECTS) $(UTOBJECTS) $(CLOBJECTS) $(LIBS)

$(TARGET_11): $(IOOBJECTS) $(COOBJECTS) $(UTOBJECTS) $(FSOBJECTS) $(CLOBJECTS) $(TARGET_11).o
	$(CPP) $(LDFLAGS) -o ./bin/$(TARGET_11) $(TARGET_11).o $(FSOBJECTS) $(IOOBJECTS) $(COOBJECTS) $(UTOBJECTS) $(CLOBJECTS) $(METIS_LIB)

$(TARGET_12): $(IOOBJECTS) $(COOBJECTS) $(UTOBJECTS) $(FSOBJECTS) $(CLOBJECTS) $(TARGET_12).o
	$(CPP) $(LDFLAGS) -o ./bin/$(TARGET_12) $(TARGET_12).o $(FSOBJECTS) $(IOOBJECTS) $(COOBJECTS) $(UTOBJECTS) $(CLOBJECTS) $(METIS_LIB)


.c.o:
	$(CC) $(CFLAGS) -c $< -I$(INCLUDES) -o $@

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $< -I$(INCLUDES) -o $@

$(IOFOLDER)/%.o: $(IOFOLDER)/%.cpp
	$(CPP) $(CPPFLAGS) -c $< -I$(INCLUDES) -o $@

$(COFOLDER)/%.o: $(COFOLDER)/%.cpp
	$(CPP) $(CPPFLAGS) -c $< -I$(INCLUDES) -o $@

$(UTFOLDER)/%.o: $(UTFOLDER)/%.cpp
	$(CPP) $(CPPFLAGS) -c $< -I$(INCLUDES) -o $@


clean:
	rm -f $(TARGET_1).o $(TARGET_2).o $(TARGET_3).o $(FSFOLDER)/*.o $(IOFOLDER)/*.o $(COFOLDER)/*.o $(UTFOLDER)/*.o $(CLFOLDER)/*.o blosc_filter/*.so

#wipe:
#	rm -f $(TARGET).o $(OBJECTS) $(TARGET) *~ *.bak

message:
	echo "Executables: " $(TARGET) " have been created"

print-%: ; @echo $*=$($*)


#HDF5 PARSER
BLOSC_FILTER_ROOT=blosc_filter
BLOSC_FILTER_LIB=$(BLOSC_FILTER_ROOT)/blosc_filter.so
HDF_INC=$(HDF5_ROOT)/include
HDF_LIB=$(HDF5_ROOT)/lib

IOFILESH5 = $(wildcard $(IOFOLDER)/*.cc)
IOOBJECTSH5 = $(addprefix $(IOFOLDER)/,$(notdir $(IOFILESH5:.cc=.o)))

$(TARGET_H5): $(BLOSC_FILTER_LIB) $(IOOBJECTS) $(IOOBJECTSH5) $(COOBJECTS) $(UTOBJECTS) $(FSOBJECTS) $(CLOBJECTS) $(TARGET_4).o 
	$(CPP) -DUSEHDF5 $(LDFLAGS) -o ./bin/$(TARGET_H5) $(TARGET_4).o $(BLOSC_FILTER_LIB) $(FSOBJECTS) \
	$(IOOBJECTS) $(IOOBJECTSH5) $(COOBJECTS) $(UTOBJECTS) $(CLOBJECTS) $(LIBS) -L$(BLOSC_ROOT)/lib -lblosc -L$(HDF_LIB) -lhdf5 -lhdf5_cpp

$(IOFOLDER)/%.o: $(IOFOLDER)/%.cc
	$(CPP) $(CPPFLAGS) -c $< -I$(INCLUDES) -I$(HDF_INC) -I$(BLOSC_ROOT)/include -I$(BLOSC_FILTER_ROOT) -o $@

$(BLOSC_FILTER_LIB):
	make -C $(BLOSC_FILTER_ROOT)
