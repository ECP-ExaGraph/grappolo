# Grappolo: Parallel clustering using the Louvain method as the serial template

## Description

Grappolo implements a parallel version of the Louvain community detection algorithm, using several heuristics to gain computational speed. There can be a larger memory footprint and some loss of accuracy arising from non-deterministic order of vertex processing and use of different heuristics. In general, we have observed significant gains in speed with minimal impact on clustering accuracy (measure in terms of the final modularity score). Further, Grappolo enables processing of large inputs that would otherwise remain unsolved using the serial Louvain implementation. We also note that the distributed-memory version (Vite) is available for extremely large data sets.


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





## Input parameters that can be customized:

`links` A numeric matrix of network edges.

`coloring` (1) An integer between 0 and 3 that controls the distance-1 graph coloring heuristic used to partition independent sets of vertices in a graph for parallel processing. 
  * 0 - No coloring.
  * 1 - (Default) Distance-1 graph coloring. Every vertex receives a color and no two neighbors have the same color.
  * 2 – Distance-1 graph coloring, rebalanced for evenly distributed color classes (#nodes per color).
  * 3 - Incomplete coloring, limited to `numColors`, by default 16.

`numColors` (16): An integer between 1 and 1024. Limits graph coloring. Only used if `coloring=3`, incomplete coloring, is set.

`C_thresh` (1e-6): The threshold value determines how long the algorithm iterates. This value (a real number between 0 and 1; >0) is checked when coloring is enabled. The algorithm will stop iterating when the gain in modularity becomes less than `C_thresh`. A final iteration is performed using the value specified by the `threshold` parameter. Desired value for `C_thresh` should be larger than `threshold` for gains in performance.

`minGraphSize` (1,000): Determines when multi-phase operations should stop. Execution stops when the coarsened graph has collapsed the current graph to a fewer than `minGraphSize` nodes. 

`threshold` (1e-9): The threshold value determines how long the algorithm iterates. It is a real number between 0 and 1 (>0). The algorithm will stop the iterations in the current phase when the gain in modularity is less than `threshold`. The algorithm can enter the next phase based on the size of the coarsened graph. 

`syncType` (0) An integer between 0 and 4 that controls synchronization between threads. Only applies if `coloring=0` (no coloring). Synchronization forces the parallel algorithm to behave similar to the execution of a serial Louvain implementation.
  * 0 - (Default) No sync (gives the best performance in terms of runtime).
  * 1 - Full sync (behaves like a serial algorithm).
  * 2 - Neighborhood sync (a hybrid between 0 and 1).
  * 3 - Early termination (stops processing a vertex if it has not changed its community for the past few iterations – leads to gain in performance).
  * 4 - Full sync with early termination (hybrid of 1 and 3).

`basicOpt` (1) Either 0 or 1, controls the representation of intermediate data structures.
  * 0 – Uses stl::map data structure. While it has a smaller memory footprint and computational efficiency, excessive memory allocations and deallocations can lead to loss of performance.
  * 1 - (Default) Use stl::vector to replace the functionality of stl::map. This option comes at the expense of a larger memory footprint and loss in performance when the algorithm has a large number of communities and does not converge quickly (does not have a good community structure). In general, this option can be faster than the stl::map option for inputs with good community structure.


## Further Details:

The only required parameter is `links`. All other parameters are tuning parameters that control how speed vs accuracy vs memory tradeoffs are made. We specifically note that the original Louvain algorithm is non-deterministic and varies considerably for different input structures. Grappolo inherits these limitations with added complications from parallelization.

## Return Value:

A list with two elements:

  * `modularity` - The modularity of the computed partitioning of the network into a set of non-overlapping clusters (communities or partitions). Modularity is a measure of the connectedness in a given network when partitioned based on a given approach.
  * `communities` - A vector where the i'th value is the cluster number that the i'th node in the links matrix has been assigned to.



