// ***********************************************************************
//
//            Grappolo: A C++ library for graph clustering
//               Mahantesh Halappanavar (hala@pnnl.gov)
//               Pacific Northwest National Laboratory
//
// ***********************************************************************
//
//       Copyright (2014) Battelle Memorial Institute
//                      All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************

#include "defs.h"
#include "input_output.h"
#include "basic_comm.h"
#include "basic_util.h"
#include "utilityClusteringFunctions.h"
#include "color_comm.h"
#include "sync_comm.h"

#ifdef USE_PMEM_ALLOC
static char path[PATH_MAX]="/pmem1";
#endif

using namespace std;


//#define USEHDF5 
int main(int argc, char** argv) {
    
    //Parse Input parameters:
    clustering_parameters opts;
    if (!opts.parse(argc, argv)) {
        return -1;
    }
    int nT = 1; //Default is one thread
    bool edgeListBinaryToPMEM = false;
#pragma omp parallel
    {
        nT = omp_get_num_threads();
    }
    if (nT < 1) {
        printf("The number of threads should be greater than one.\n");
        return 0;
    }
    
    // File Loading
    double time1, time2;
    graph* G = (graph *) malloc (sizeof(graph));
    
    /* Step 2: Parse the graph in Matrix Market format */
    int fType = opts.ftype; //File type
    char *inFile = (char*) opts.inFile;
    switch (fType) {
        case 1: parse_MatrixMarket_Sym_AsGraph(G, inFile); break;
        case 2: parse_Dimacs9FormatDirectedNewD(G, inFile); break;
        case 3: parse_PajekFormat(G, inFile); break;
        case 4: parse_PajekFormatUndirected(G, inFile); break;
        case 5: loadMetisFileFormat(G, inFile); break;
        case 6: parse_UndirectedEdgeList(G, inFile); break;
            //parse_UndirectedEdgeListDarpaHive(G, inFile); break;
        case 7: printf("This routine is under development.\n"); exit(1); break;
                /*parse_DirectedEdgeList(G, inFile); break;*/
        case 8: parse_SNAP(G, inFile); break;
        case 9: parse_EdgeListBinaryNew(G,inFile); break;
        case 10:
#ifdef USEHDF5                
            //parse_EdgeListCompressedHDF5(G,inFile);
            parse_EdgeListCompressedHDF5NoDuplicates(G,inFile);
#endif
                break;
        case 11: parse_UndirectedEdgeListFromJason(G, inFile); break;
        case 12: parse_UndirectedEdgeListWeighted(G, inFile); break; // for John F's graphs
        case 13: parse_UndirectedEdgeListDarpaHive(G, inFile); break;
        case 14: edgeListBinaryToPMEM = true; break;
        case 15: parse_EdgeListFromGorder(G, inFile); break;
        default:  cout<<"A valid file type has not been specified"<<endl; exit(1);
    }
    int threadsOpt = 0;
    if(opts.threadsOpt)
	    threadsOpt =1;
    threadsOpt =1;

    int replaceMap = 0;
    if(  opts.basicOpt == 1 )
	    replaceMap = 1;

    long NV, NE, weighted;
    long *C_orig = NULL;

    // directly read graph into PMEM without duplication
    if (edgeListBinaryToPMEM) {
	    
	    // read #vertices and #edges
	    std::ifstream ifs;  
	    ifs.open(inFile, std::ifstream::in | std::ifstream::binary);
	    if (!ifs) {
		    std::cerr << "Error opening binary format file: " << inFile << std::endl;
		    exit(EXIT_FAILURE);
	    }
	    //Parse line-1: #Vertices #Edges
	    ifs.read(reinterpret_cast<char*>(&NV), sizeof(NV));
	    ifs.read(reinterpret_cast<char*>(&NE), sizeof(NE));
	    ifs.read(reinterpret_cast<char*>(&weighted), sizeof(weighted));
            G->numVertices = NV;
            G->sVertices = NV;
            G->numEdges = NE;
#ifdef USE_PMEM_ALLOC
	    // create PMEM partition with specific size
#ifdef USE_FS_DAX
	    int err = memkind_create_pmem(path, PMEM_MAX_SIZE, &pmem_kind);
	    if (err) {
		    printf("Error while creating pmem_kind.\n");    
		    print_err_message(err);
		    exit(1);
	    }
#endif    
#ifdef DO_PMEM_DEBUG
	    size_t resident, active, allocated;
	    memkind_update_cached_stats();
#ifdef USE_FS_DAX
	    memkind_get_stat(pmem_kind, MEMKIND_STAT_TYPE_RESIDENT, &resident);
	    memkind_get_stat(pmem_kind, MEMKIND_STAT_TYPE_ACTIVE, &active);
	    memkind_get_stat(pmem_kind, MEMKIND_STAT_TYPE_ALLOCATED, &allocated);
#else
	    memkind_get_stat(NULL, MEMKIND_STAT_TYPE_RESIDENT, &resident);
	    memkind_get_stat(NULL, MEMKIND_STAT_TYPE_ACTIVE, &active);
	    memkind_get_stat(NULL, MEMKIND_STAT_TYPE_ALLOCATED, &allocated);
#endif
	    printf ("PMEM stats: resident=%zu active=%zu allocated=%zu\n", resident, active, allocated);
#endif   
#ifdef USE_FS_DAX
	    G->edgeListPtrs = (long *)memkind_malloc(pmem_kind, ((NV+1)*sizeof(long)));
#elif USE_KI_DAX
	    G->edgeListPtrs = (long *)memkind_malloc(MEMKIND_DAX_KMEM_INTERLEAVE, ((NV+1)*sizeof(long)));
#else
	    G->edgeListPtrs = (long *)memkind_malloc(MEMKIND_DAX_KMEM_PREFERRED, ((NV+1)*sizeof(long)));
#endif
	    if (G->edgeListPtrs == NULL) {
		    fprintf(stderr, "Exiting after unable to allocate PMEM for G->edgeListPtrs of size: %zu.\n", (NV+1)*sizeof(long));
		    exit(1);
	    }
#ifdef USE_FS_DAX
	    G->edgeList = (edge *)memkind_malloc(pmem_kind, (2*NE*sizeof(edge)));
#elif USE_KI_DAX
	    G->edgeList = (edge *)memkind_malloc(MEMKIND_DAX_KMEM_INTERLEAVE, (2*NE*sizeof(edge)));
#else
	    G->edgeList = (edge *)memkind_malloc(MEMKIND_DAX_KMEM_PREFERRED, (2*NE*sizeof(edge)));
#endif
	    if (G->edgeList == NULL) {
		    fprintf(stderr, "Exiting after unable to allocate PMEM for G->edgeList of size: %zu.\n", 2*NE*sizeof(edge));
		    exit(1);
	    }
	    printf("PMEM allocation successful, allocated graph CSR to PMEM (in bytes): %zu.\n", ((NV+1)*sizeof(long) + 2*NE*sizeof(edge)));
#else
	    G->edgeListPtrs = (long *)  malloc((NV+1) * sizeof(long));
	    assert(G->edgeListPtrs != NULL);
	    //WARNING: There is a bug in edge counting when self-loops exist
	    G->edgeList = (edge *) malloc( 2*NE * sizeof(edge));
	    assert( G->edgeList != NULL);
#endif

	    // read data from binary file directly into PMEM    
            // vertex pointers
	    ifs.read(reinterpret_cast<char*>(G->edgeListPtrs), sizeof(long) * (NV+1));	    
	    
	    // store the edge list offsets
#if defined(GRAPH_FT_LOAD)
	    long currpos = ifs.tellg();
	    long idx = 0;
	    long* vidx = (long*) malloc(NV * sizeof(long));

	    // ---NEW---
	    const int num_sockets = (GRAPH_FT_LOAD == 0) ? 1 : GRAPH_FT_LOAD;
	    printf("Read file from %d sockets\n", num_sockets);
	    int n_blocks = num_sockets;

	    long NV_blk_sz = NV / n_blocks;
	    long tid_blk_sz = omp_get_num_threads() / n_blocks;

	    // For each block
#pragma omp parallel
	    {
		    for (int b=0; b<n_blocks; b++) {

			    long NV_beg = b * NV_blk_sz;
			    long NV_end = std::min(NV, ((b+1) * NV_blk_sz) );
			    int tid_doit = b * tid_blk_sz;

			    if (omp_get_thread_num() == tid_doit) {
				    // For each vertex within block
				    for (long i = NV_beg; i < NV_end ; i++) {
					    // Ensure first-touch allocation
					    // read and initialize using your code

					    vidx[i] = idx;
					    const long vcount = G->edgeListPtrs[i+1] - G->edgeListPtrs[i];
					    idx += vcount;
					    ifs.seekg(currpos + vidx[i] * sizeof(edge), ios::beg);
					    ifs.read(reinterpret_cast<char*>(&G->edgeList[vidx[i]]), sizeof(edge) * (vcount));
				    }
			    }
		    }
	    }
	    free(vidx);
#else 
	    ifs.read(reinterpret_cast<char*>(G->edgeList), sizeof(edge) * (2*NE));
#endif
            ifs.close(); //Close the file

	    displayGraphCharacteristics(G);
	    // allocate communities
	    C_orig = (long *) malloc (NV * sizeof(long)); assert(C_orig != 0);
#if defined(RUN_FIRST_PHASE)
#else
#ifdef USE_PMEM_ALLOC
	    printf("PMEM support is experimental and tested only for a single phase currently.\n");
#endif
	    graph* G_orig = (graph *) malloc (sizeof(graph)); //The original version of the graph
	    duplicateGivenGraph(G, G_orig);    
#endif
	    if( opts.VF ) {
		   printf("Exiting: The chosen option conflicts with enabling direct loading of graph into PMEM.\n");
		   exit(1);
	    }
    }
    else {

	    displayGraphCharacteristics(G);
	    
	    /* Vertex Following option */
	    if( opts.VF ) {
		    printf("Vertex following is enabled.\n");
		    time1 = omp_get_wtime();
		    long numVtxToFix = 0; //Default zero
		    long *C = (long *) malloc (G->numVertices * sizeof(long)); assert(C != 0);
		    numVtxToFix = vertexFollowing(G,C); //Find vertices that follow other vertices
		    if( numVtxToFix > 0) {  //Need to fix things: build a new graph
			    printf("Graph will be modified -- %ld vertices need to be fixed.\n", numVtxToFix);
			    graph *Gnew = (graph *) malloc (sizeof(graph));
			    long numClusters = renumberClustersContiguously(C, G->numVertices);
			    buildNewGraphVF(G, Gnew, C, numClusters);
			    //Get rid of the old graph and store the new graph
			    free(G->edgeListPtrs);
			    free(G->edgeList);
			    free(G);
			    G = Gnew;
		    }
		    free(C); //Free up memory
		    printf("Graph after modifications:\n");
		    displayGraphCharacteristics(G);
	    }//End of if( VF == 1 )

	    // Datastructures to store clustering information
	    NV = G->numVertices;
	    C_orig = (long *) malloc (NV * sizeof(long)); assert(C_orig != 0);
	    graph* G_orig = (graph *) malloc (sizeof(graph)); //The original version of the graph
	    duplicateGivenGraph(G, G_orig);
    }  

    //Call the clustering algorithm:
    //Call the clustering algorithm:
    if ( opts.strongScaling ) { //Strong scaling enabled
        //Retain the original copy of the graph:
        graph* G_original = (graph *) malloc (sizeof(graph)); //The original version of the graph
        time1 = omp_get_wtime();
        duplicateGivenGraph(G, G_original);
        time2 = omp_get_wtime();
        printf("Time to duplicate : %lf\n", time2-time1);
        
        //Run the algorithm in powers of two for the maximum number of threads available
        int curThread = 2; //Start with two threads
        while (curThread <= nT) {
            printf("\n\n***************************************\n");
            printf("Starting run with %d threads.\n", curThread);
            printf("***************************************\n");
            //Call the clustering algorithm:
#pragma omp parallel for
            for (long i=0; i<G->numVertices; i++) {
                C_orig[i] = -1;
            }
            if(opts.coloring != 0){
                runMultiPhaseColoring(G, C_orig, opts.coloring, opts.numColors, replaceMap, opts.minGraphSize, opts.threshold, opts.C_thresh, curThread, threadsOpt);
            }else if(opts.syncType != 0){
                runMultiPhaseSyncType(G, C_orig, opts.syncType, opts.minGraphSize, opts.threshold, opts.C_thresh, curThread, threadsOpt);
            }else{
#if defined(RUN_FIRST_PHASE)
                runMultiPhaseBasicOnceWithoutRebuilding(G, C_orig, opts.syncType, opts.minGraphSize, opts.threshold, opts.C_thresh, curThread, threadsOpt);
#else                
		runMultiPhaseBasic(G, C_orig, opts.basicOpt, opts.minGraphSize, opts.threshold, opts.C_thresh, curThread,threadsOpt);
#endif
            }

            //Increment thread and revert back to original graph
            if (curThread < nT) {
                //Skip copying at the very end
                //Old graph is already destroyed in the above function
                G = (graph *) malloc (sizeof(graph)); //Allocate new space
                duplicateGivenGraph(G_original, G); //Copy the original graph to G
            }
            curThread = curThread*2; //Increment by powers of two
        }//End of while()
    } else { //No strong scaling -- run once with max threads

#pragma omp parallel for
        for (long i=0; i<NV; i++) {
            C_orig[i] = -1;
        }
        if(opts.coloring != 0){
            runMultiPhaseColoring(G, C_orig, opts.coloring, opts.numColors, replaceMap, opts.minGraphSize, opts.threshold, opts.C_thresh, nT, threadsOpt);
        }else if(opts.syncType != 0){
            runMultiPhaseSyncType(G, C_orig, opts.syncType, opts.minGraphSize, opts.threshold, opts.C_thresh, nT,threadsOpt);
        }else{
#if defined(RUN_FIRST_PHASE)
            runMultiPhaseBasicOnceWithoutRebuilding(G, C_orig, opts.syncType, opts.minGraphSize, opts.threshold, opts.C_thresh, nT, threadsOpt);
#else                
            runMultiPhaseBasic(G, C_orig, opts.basicOpt, opts.minGraphSize, opts.threshold, opts.C_thresh, nT,threadsOpt);
#endif
        }
    }
    
    //Check if cluster ids need to be written to a file:
    if( opts.output ) {
        char outFile[256];
        sprintf(outFile,"%s_clustInfo", opts.inFile);
        printf("Cluster information will be stored in file: %s\n", outFile);
        FILE* out = fopen(outFile,"w");
        for(long i = 0; i<NV;i++) {
            fprintf(out,"%ld\n",C_orig[i]);
        }
        fclose(out);
    }
    
    /*
    //Cluster Analysis:
    //Count the frequency of colors:
    long numClusters = 0;
    for(long i = 0; i < NV; i++) {
        if(C_orig[i] > numClusters)
            numClusters = C_orig[i];
    }
    numClusters++;
    printf("********************************************\n");
    printf("Number of clusters = %ld\n", numClusters);
    printf("********************************************\n");
    long *colorFreq = (long *) malloc (numClusters * sizeof(long)); assert(colorFreq != 0);
#pragma omp parallel for
    for(long i = 0; i < numClusters; i++) {
        colorFreq[i] = 0;
    }
#pragma omp parallel for
    for(long i = 0; i < NV; i++) {
        assert(C_orig[i] < numClusters);
        if(C_orig[i] >= 0) {
            __sync_fetch_and_add(&colorFreq[C_orig[i]],1);
        }
    }
    for(long i=0; i < numClusters; i++) {
        printf("%ld \t %ld\n", i, colorFreq[i]);
    }
    printf("********************************************\n");
    double gini = computeGiniCoefficient(colorFreq, numClusters);
    printf("Gini coefficient      : %lf\n", gini);
    printf("********************************************\n");
    free(colorFreq);
    
    char outFile[256];
    sprintf(outFile,"%s_PajekWithComm.net", opts.inFile);
    writeGraphPajekFormatWithCommunityInfo(G_orig, outFile, C_orig);
    */
    //Cleanup:
    if(C_orig != 0) free(C_orig);
    //Do not free G here -- it will be done in another routine.
    
    return 0;
}//End of main()
