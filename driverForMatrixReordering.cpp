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

using namespace std;
//#define USEHDF5 
int main(int argc, char** argv) {
    
    //Parse Input parameters:
    clustering_parameters opts;
    if (!opts.parse(argc, argv)) {
        return -1;
    }
    int nT = 1; //Default is one thread
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
    bool isSym = true; //Assume symmetric by default
    switch (fType) {
        case 1: //parse_MatrixMarket_Sym_AsGraph(G, inFile); break;
            isSym = parse_MatrixMarket(G, inFile); break;
        case 2: parse_Dimacs9FormatDirectedNewD(G, inFile); break;
        case 3: parse_PajekFormat(G, inFile); break;
        case 4: parse_PajekFormatUndirected(G, inFile); break;
        case 5: loadMetisFileFormat(G, inFile); break;
        case 6: //parse_UndirectedEdgeList(G, inFile);
            parse_UndirectedEdgeListDarpaHive(G, inFile); break;
        case 7: /* parse_DirectedEdgeList(G, inFile); break; */
            printf("This routine is under development.\n"); exit(1); break;
        case 8: parse_SNAP(G, inFile); break;
        case 9: parse_EdgeListBinaryNew(G,inFile); break;
        case 10:
#ifdef USEHDF5                
            //parse_EdgeListCompressedHDF5(G,inFile);
            parse_EdgeListCompressedHDF5NoDuplicates(G,inFile);
#endif
            break;
        default:  cout<<"A valid file type has not been specified"<<endl; exit(1);
    }
    
    displayGraphCharacteristics(G);
    int threadsOpt = 0;
    if(opts.threadsOpt)
        threadsOpt =1;
    threadsOpt =1;
    
    int replaceMap = 0;
    if(  opts.basicOpt == 1 )
        replaceMap = 1;
    
    /* Vertex Following option */
    if( opts.VF ) {
        printf("Vertex following is not supported for this exercise.\n");
    }//End of if( VF == 1 )
    
    // Datastructures to store clustering information
    long NV = G->numVertices;
    long NS = G->sVertices;
    long NT = NV - NS;
    long *C_orig = (long *) malloc (NV * sizeof(long)); assert(C_orig != 0);
    
    //Retain the original copy of the graph:
    graph* G_original = (graph *) malloc (sizeof(graph)); //The original version of the graph
    time1 = omp_get_wtime();
    duplicateGivenGraph(G, G_original);
    time2 = omp_get_wtime();
    printf("Time to duplicate : %lf\n", time2-time1);
    
    //Call the clustering algorithm:
    //Call the clustering algorithm:
    if ( opts.strongScaling ) { //Strong scaling enabled
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
                runMultiPhaseColoring(G, C_orig, opts.coloring, opts.numColors, replaceMap, opts.minGraphSize, opts.threshold, opts.C_thresh, nT, threadsOpt);
            }else if(opts.syncType != 0){
                runMultiPhaseSyncType(G, C_orig, opts.syncType, opts.minGraphSize, opts.threshold, opts.C_thresh, nT,threadsOpt);
            }else{
                runMultiPhaseBasic(G, C_orig, opts.basicOpt, opts.minGraphSize, opts.threshold, opts.C_thresh, nT,threadsOpt);
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
            runMultiPhaseBasic(G, C_orig, opts.basicOpt, opts.minGraphSize, opts.threshold, opts.C_thresh, nT,threadsOpt);
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
    
    //Now build the old2New vertex map:
    printf("About to build the old2NewMap. \n");
    long *commIndex  = (long *) malloc (NV * sizeof(long)); assert(commIndex != 0);
    long *old2NewMap = (long *) malloc (NV * sizeof(long)); assert(old2NewMap != 0);
#pragma omp parallel for
    for (long i=0; i<NV; i++) { //initialize
        commIndex[i] = -1;
        old2NewMap[i] = -1;
    }
    
    buildOld2NewMap(NV, C_orig, commIndex); //Pack the vertices based on community detection
    char outFileMat[256];
    sprintf(outFileMat,"%s_Reordered.mtx", opts.inFile); //Output file name
    //Segregate the vertices in case it is a bipartite graph
    if(isSym) {
#pragma omp parallel for
        for (long i=0; i<NV; i++) {
            old2NewMap[commIndex[i]] = i;
        }
        writeGraphMatrixMarketFormatSymmetricReordered(G_original, outFileMat, old2NewMap);
    } else {
        //STEP 1: Segregate the row and column vertices
        long rowCounter = 0;
        long colCounter = NS;
        long *Rprime    = (long *) malloc (NV * sizeof(long)); assert(Rprime != 0);
        for (long i=0; i<NV; i++) {
            Rprime[i]= -1;
        }
        for (long i=0; i<NV; i++) { //Go through the list in a reverse order
            if(commIndex[i] < NS) { //A row vertex
                Rprime[rowCounter] = commIndex[i];
                rowCounter++;
            } else { //A column vertex
                Rprime[colCounter] = commIndex[i];
                colCounter++;
            }
        }//End of for(i)
        assert(rowCounter==NS); assert(colCounter==NV); //Sanity check
        //STEP 2: Now build the old2New map:
        for (long i=0; i<NV; i++) {
            old2NewMap[Rprime[i]] = i; //pOrder is a old2New index mapping
        }
        //Clean up:
        free(Rprime);
        //STEP 3: Write the bipartite graph to a file
        writeGraphMatrixMarketFormatBipartiteReordered(G_original, outFileMat, old2NewMap);
    }
    
    //Cleanup:
    if(C_orig != 0) free(C_orig);
    if(commIndex != 0) free(commIndex);
    if(old2NewMap != 0) free(old2NewMap);
    if(G_original != 0) {
        free(G_original->edgeListPtrs);
        free(G_original->edgeList);
        free(G_original);
    }
    
    return 0;
}//End of main()
