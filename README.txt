

The single slice Grappolo has been divided to 7 folders

/DefineStructure: Contain all .h files from different dirctories
/Utility: check basic_ultil.h and utilityClusteringFunc.h
/BasicCommunitiesDection: check basic_comm.h
/Coloring: check coloring.h and comm_coloring.h
/FullSyncOptimization: check sync_comm.h
/InputsOutput: check input_output.h


/****************************************************/
Makefile will create 3 executable in the /bin folder
	1)	./convertFileToBinary
	2)	./driverForGraphClustering
	3)	./driverForColoring



/****************************************************/
To update code, record each update in the folder.
To updates for each particular type of communities detection

1) Change code in particualr folder
	/ Add different communities detection method
	
2) Change the runMultiPhaseXXX.cpp to capture the changes

3) Update the .h files in /DefineStructure

4) Drivers and other folder can remain unchanged

5) To update the Utility code must be done with care, API should
	stay the same

/****************************************************/
To run the code, it will be in the menu of ./driverForGraphClustering
