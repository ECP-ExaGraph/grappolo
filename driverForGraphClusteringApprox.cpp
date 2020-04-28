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
#include <numeric>

using namespace std;

int main(int argc, char **argv)
{
    
    //Parse Input parameters:
    clustering_parameters opts;
    if (!opts.parse(argc, argv))
    {
        return -1;
    }
    int nT = 1; //Default is one thread
    if (!opts.compute_metrics)
    {
#pragma omp parallel
        {
            nT = omp_get_num_threads();
        }
        if (nT <= 1)
        {
            printf("The number of threads should be greater than one.\n");
            return 0;
        }
    }
    else
    {
        omp_set_num_threads(1);
    }
    printf("The number of threads %d.\n", omp_get_num_threads());
    
    // File Loading
    double time1, time2;
    graph *G = (graph *)malloc(sizeof(graph));
    
    /* Step 2: Parse the graph in Matrix Market format */
    int fType = opts.ftype; //File type
    char *inFile = (char *)opts.inFile;
    if (fType == 1)
        parse_MatrixMarket_Sym_AsGraph(G, inFile);
    else if (fType == 2)
        parse_Dimacs9FormatDirectedNewD(G, inFile);
    else if (fType == 3)
        parse_PajekFormat(G, inFile);
    else if (fType == 4)
        parse_PajekFormatUndirected(G, inFile);
    else if (fType == 5)
        loadMetisFileFormat(G, inFile);
    else if (fType == 6)
        parse_UndirectedEdgeList(G, inFile);
    else if (fType == 7)
        parse_DirectedEdgeList(G, inFile);
    else if (fType == 8)
        parse_SNAP(G, inFile);
    else if (fType == 9)
        parse_EdgeListBinaryNew(G, inFile);
    else
    {
        cout << "Not a valid file type" << endl;
        exit(1);
    }
    
    //long percentange = 80;
    displayGraphCharacteristics(G);
    
    int threadsOpt = 0;
    if (opts.threadsOpt)
        threadsOpt = 1;
    threadsOpt = 1;
    
    int replaceMap = 0;
    if( opts.replaceMap )
        replaceMap = 1;
    
    
    if (opts.compute_metrics)
    {
        omp_set_num_threads(1);
        const int NUM_FILES = 10;
        std::vector<int> num_matches(G->numEdges, 0);
        const long nv = G->numVertices;
        
        //std::string inputFileName = inFile;
        std::string inputFileName(inFile);
        std::cerr << inputFileName << std::endl;
        std::string base_filename = inputFileName.substr(inputFileName.find_last_of("/\\") + 1);
        std::string::size_type const p(base_filename.find_last_of('.'));
        std::string graphname = base_filename.substr(0, p);
        
        std::vector<int> vtx_to_community(nv, 0);
        
        const long ne = G->numEdges;
        std::vector<long> edgeCount(nv + 1L);
        std::vector<std::pair<long, long>> edgeList(ne);
        
        for (long int i = 0L; i < ne; i++)
        {
            edgeList[i].first = G->edgeList[i].head;
            edgeList[i].second = G->edgeList[i].tail;
            edgeCount[edgeList[i].first + 1L]++;
        }
        
        std::vector<long> ecTmp(nv + 1L);
        std::vector<long> edgeListIndexes(nv + 1L);
        
        std::partial_sum(edgeCount.begin(), edgeCount.end(), ecTmp.begin());
        edgeCount = ecTmp;
        
        edgeListIndexes[0] = 0;
        
        for (long i = 0L; i < nv; i++)
            edgeListIndexes[i + 1L] = edgeCount[i + 1L];
        
        edgeCount.resize(0L);
        
        auto ecmp = [](const std::pair<long, long> &e0, const std::pair<long, long> &e1)
        { return ((e0.first < e1.first) || ((e0.first == e1.first) && (e0.second < e1.second))); };
        
        if (!std::is_sorted(edgeList.begin(), edgeList.end(), ecmp))
        {
            //std::cout << "Edge list is not sorted" << std::endl;
            std::sort(edgeList.begin(), edgeList.end(), ecmp);
        }
        else
            std::cout << "Edge list is sorted!" << std::endl;
        
        for (int f = 1; f < NUM_FILES + 1; f++)
        {
            
            std::ifstream ifs;
            if (opts.coloring)
                ifs.open("./results/" + graphname + ".color.rand.out" + std::to_string(f), std::ifstream::in);
            else
            {
                std::string rfile = "./results/" + graphname + ".nocolor.rand.out" + std::to_string(f);
                std::cout << rfile << std::endl;
                ifs.open(rfile, std::ifstream::in);
            }
            if (!ifs)
            {
                std::cerr << "Error opening output file: "
                << "results/" + graphname + ".{no}color.rand.out" + std::to_string(f) << std::endl;
                exit(EXIT_FAILURE);
            }
            // std::cerr << "The output file: "
            //           << "results/" + graphname + ".{no}color.rand.out" + std::to_string(f) << std::endl;
            
            for (long i = 0; i < nv; i++)
            {
                ifs >> vtx_to_community[i];
            }
            ifs.close();
            long c = 0;
            //#pragma omp parallel for default(none), private(nv), shared(g, vtx_to_community), schedule(guided)
            for (long i = 0; i < nv; i++)
            {
                long e0 = edgeListIndexes[i];
                long e1 = edgeListIndexes[i + 1];
                
                for (long k = e0; k < e1; k++, c++)
                {
                    assert(i == edgeList[k].first);
                    if (vtx_to_community[i] == vtx_to_community[edgeList[k].second])
                    {
                        num_matches[c] += 1;
                    }
                }
            }
        }
        std::vector<int> histogram(11, 0);
        long non_zero = 0;
        for (int e = 0; e < G->numEdges; e++)
        {
            //std::cout<<num_matches[e]/NUM_FILES<<std::endl;
            if (num_matches[e] > 0)
                non_zero++;
            histogram[(int)(num_matches[e] * 10.0 / NUM_FILES)] += 1;
        }
        std::cout << "Num edges with non-zero matches=" << non_zero << std::endl;
        std::cout << "Total number of edges=" << G->numEdges << std::endl;
        std::cout << "Per edge match ratio (mean) = " << non_zero * 1.0 / G->numEdges << std::endl;
        std::cout << "Max matches per edge=" << *std::max_element(num_matches.begin(), num_matches.end()) << std::endl;
        std::cout << "histogram:" << std::endl;
        for (auto h : histogram)
        {
            std::cout << h * 1.0 / G->numEdges << std::endl;
        }
        exit(1);
    }
    
    /* Vertex Following option */
    if (opts.VF)
    {
        printf("Vertex following is enabled.\n");
        time1 = omp_get_wtime();
        long numVtxToFix = 0; //Default zero
        long *C = (long *)malloc(G->numVertices * sizeof(long));
        assert(C != 0);
        numVtxToFix = vertexFollowing(G, C); //Find vertices that follow other vertices
        if (numVtxToFix > 0)
        { //Need to fix things: build a new graph
            printf("Graph will be modified -- %ld vertices need to be fixed.\n", numVtxToFix);
            graph *Gnew = (graph *)malloc(sizeof(graph));
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
    } //End of if( VF == 1 )
    
    // Datastructures to store clustering information
    long NV = G->numVertices;
    long *C_orig = (long *)malloc(NV * sizeof(long));
    assert(C_orig != 0);
    
    //Call the clustering algorithm:
    if (opts.strongScaling)
    {
        //Strong scaling enabled
        printf("Scaling is not support in the current version\n");
    }
    
#pragma omp parallel for
    for (long i = 0; i < NV; i++)
    {
        C_orig[i] = -1;
    }
    
    //runMultiPhaseLouvainAlgorithm(G, C_orig, coloring, replaceMap, opts.minGraphSize, opts.threshold, opts.C_thresh, nT,threadsOpt);
    // Change to each sub function that belong to the folder
    if (opts.coloring != 0)
    {
        //runMultiPhaseColoring(G, C_orig, opts.coloring, opts.minGraphSize, opts.threshold, opts.C_thresh, nT, threadsOpt);
        runMultiPhaseColoring(G, C_orig, opts.coloring, opts.numColors, replaceMap, opts.minGraphSize, opts.threshold, opts.C_thresh, nT, threadsOpt);
    }
    else if (opts.syncType != 0)
    {
        runMultiPhaseSyncType(G, C_orig, opts.syncType, opts.minGraphSize, opts.threshold, opts.C_thresh, nT, threadsOpt);
    }
    else
    {
        //runMultiPhaseBasic(G, C_orig, opts.basicOpt, opts.minGraphSize, opts.threshold, opts.C_thresh, nT,threadsOpt);
        runMultiPhaseBasicApprox(G, C_orig, opts.basicOpt, opts.minGraphSize, opts.threshold, opts.C_thresh, nT, threadsOpt, opts.percentage);
    }
    
    //Check if cluster ids need to be written to a file:
    if (opts.output)
    {
        char outFile[256];
        sprintf(outFile, "%s_clustInfo", opts.inFile);
        printf("Cluster information will be stored in file: %s\n", outFile);
        FILE *out = fopen(outFile, "w");
        for (long i = 0; i < NV; i++)
        {
            fprintf(out, "%ld\n", C_orig[i]);
        }
        fclose(out);
    }
    
    //Cleanup:
    if (C_orig != 0)
        free(C_orig);
    //Do not free G here -- it will be done in another routine.
    
    return 0;
} //End of main()
