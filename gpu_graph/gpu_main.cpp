// ***********************************************************************
//
//     Rundemanen: CUDA C++ parallel program for community detection
//   Md Naim (naim.md@gmail.com), Fredrik Manne (Fredrik.Manne@uib.no)
//                       University of Bergen
//
// ***********************************************************************
//
//       Copyright (2016) University of Bergen
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

#include "thrust/device_vector.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <algorithm>

#include"fstream"
#include "iostream"
#include "graphHOST.h"
#include "graphGPU.h"
#include "communityGPU.h"
#include"list"

#include "defs.h"
#include "input_output.h"
#include "basic_util.h"
#include "utilityClusteringFunctions.h" // added for func sumVertexDegree

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
    switch (fType) {
        case 1: parse_MatrixMarket_Sym_AsGraph(G, inFile); break;
        case 2: parse_Dimacs9FormatDirectedNewD(G, inFile); break;
        case 3: parse_PajekFormat(G, inFile); break;
        case 4: parse_PajekFormatUndirected(G, inFile); break;
        case 5: loadMetisFileFormat(G, inFile); break;
        case 6: //parse_UndirectedEdgeList(G, inFile); break;
            parse_UndirectedEdgeListDarpaHive(G, inFile); break;
        case 7: parse_DirectedEdgeList(G, inFile); break;
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
        default:  cout<<"A valid file type has not been specified"<<endl; exit(1);
    }
    
    displayGraphCharacteristics(G);
    //std::cout << "Graph Displayed above" << std::endl;
    //displayGraph(G);

    // start parsing grappolo input to GPU code object
    // first we read to GraphHOST if that works ok
    // then we read to GraphGPU object
    GraphHOST input_graph; // GPU code graph in Host Memory
    input_graph.nb_nodes = (long) G->numVertices;
    // input_graph.nb_links = (long) G->sVertices;
    input_graph.nb_links = ((long) G->numEdges) * 2;
    std::cout << "Number of Vertices: " << input_graph.nb_nodes << std::endl;
    std::cout << "Number of Edges: " << input_graph.nb_links / 2 << std::endl;
    
    long    NV        = G->numVertices;
    long    *vtxPtr   = G->edgeListPtrs;
    edge    *vtxInd   = G->edgeList;
    
    //Allocate memory:
    input_graph.degrees.resize(input_graph.nb_nodes);
    input_graph.links.resize(input_graph.nb_links);
    input_graph.weights.resize(input_graph.nb_links);
    
    //The pointer vector:
    for(long i = 0; i < input_graph.nb_nodes; i++) {
        input_graph.degrees[i] = (int)vtxPtr[i+1];
    }
    //Make sure that number of edges is accurately represented
    std::cout << "input_graph.degrees[NV]: " << input_graph.degrees[NV]
     << "  input_graph.nb_links: " << input_graph.nb_links << std::endl;
    // assert below will not hold true for all graph inputs
    // assert(input_graph.degrees[NV] == input_graph.nb_links);
    //The edge index vector
    for(long i = 0; i < input_graph.nb_links; i++) {
        input_graph.links[i] = (int) vtxInd[i].tail;
        input_graph.weights[i] = (float) vtxInd[i].weight;
    }
    //Free memory for the graph read in Grappolo:
    if(G != 0) {
        free(G->edgeListPtrs);
        free(G->edgeList);
        free(G);
    }

    // gpu code starts below
    //int type = UNWEIGHTED;
    int type = WEIGHTED;
    double threshold = 0.000001;
    //if(argc==4) threshold = atof(argv[2]);
    double binThreshold = 0.01;
    //if(argc==4) binThreshold=atof(argv[3]);
    //binThreshold=threshold;
    //Copy Graph to Device
    Community dev_community(input_graph, -1, threshold);
#if 1
    // Coloring code below
    unsigned int custom_warp_size = 32;
    dev_community.g.greedyColoring(custom_warp_size);
    std::cout << "Coloring Done" <<  std::endl;
#endif
    double cur_mod = -1.0, prev_mod = 1.0;
    bool improvement = false;
    
    std::cout << "threshold: " << threshold << " binThreshold: " << binThreshold << std::endl;
    
    //Read Prime numbers
    dev_community.readPrimes("fewprimes.txt");
    
    cudaStream_t *streams = NULL;
    int n_streams = 8;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    //clock_t clk_decision, clk_contraction;
    std::vector<clock_t> clkList_decision;
    std::vector<clock_t> clkList_contration;
    
    clock_t t1, t2, t3;
    t1 = clock();
    
    /*
     dev_community.preProcess();
     dev_community.gatherStatistics(true); // MUST be "true" to filter out isolated vertices at the beginning
     dev_community.compute_next_graph();
     dev_community.set_new_graph_as_current();
     */
    bool TEPS = true;
    bool islastRound = false;
    int szSmallComm = 100000;
    bool isGauss = true;// false;
    
    if(isGauss)
        std::cout<<"\n Update method:  Gauss–Seidel (in batch) \n";
    else
        std::cout<<"\n Update method: Jacobi\n";
    
    int stepID = 1;
    
    do {
        std::cout << "---------------Calling method for modularity optimization------------- \n";
        t2 = clock();
        prev_mod = cur_mod;
        
        cur_mod = dev_community.one_levelGaussSeidel(cur_mod, islastRound,
                                                     szSmallComm, binThreshold, isGauss &&(dev_community.community_size > szSmallComm),
                                                     streams, n_streams, start, stop);
        
        t2 = clock() - t2;
        
        clkList_decision.push_back(t2); // push the clock for the decision
        
        std::cout<< "step: " <<stepID <<", Time for modularity optimization: " << ((float) t2) / CLOCKS_PER_SEC << std::endl;
        stepID++;
        if (TEPS == true) {
            std::cout<<binThreshold<<"_"<<threshold<< " #E:" <<  dev_community.g.nb_links << "  TEPS: " << dev_community.g.nb_links / (((float) t2) / CLOCKS_PER_SEC) << std::endl;
            TEPS = false;
        }
        
        //break;
        std::cout << "Computed modularity: " << cur_mod << " ( init_mod = " << prev_mod << " ) " << std::endl;
        
        if ((cur_mod - prev_mod) > threshold) {
            
            t2 = clock();
            t3 = t2;
            dev_community.gatherStatistics();
            t2 = clock() - t2;
            //std::cout << "T_gatherStatistics: " << ((float) t2) / CLOCKS_PER_SEC << std::endl;
            
            t2 = clock();
            dev_community.compute_next_graph(streams, n_streams, start, stop);
            t2 = clock() - t2;
            std::cout << "Time to compute next graph: " << ((float) t2) / CLOCKS_PER_SEC << std::endl;
            
            //clkList_contration.push_back(t2); // push the clock for the contraction
            
            t2 = clock();
            dev_community.set_new_graph_as_current();
            t2 = clock() - t2;
            t3 = clock() -t3;
            
            clkList_contration.push_back(t3); // push the clock for the contraction
            
            //std::cout << "T_new_graph_as_current: " << ((float) t2) / CLOCKS_PER_SEC << std::endl;
            // break;
            
            //std::cout << "\n Back to Main \n";
            //int sc;
            //std::cin>>sc;
            
            
        } else {
            if (islastRound == false) {
                islastRound = true;
            } else {
                break;
            }
        }
        //improvement = false;
    } while (true);
    
    std::cout<< "#phase: "<<stepID<<std::endl;
    
    t2 = clock();
    float diff = ((float) t2 - (float) t1);
    float seconds = diff / CLOCKS_PER_SEC;
    
    if( argc ==1){
        std::cout <<  binThreshold<<"_"<<threshold<<" Running Time: " << seconds << " ;  Final Modularity: "
        << prev_mod  << std::endl;
    }else{
        std::cout <<  binThreshold<<"_"<<threshold<<" Running Time: " << seconds << " ;  Final Modularity: "
        << prev_mod << " inputGraph: " << argv[1] << std::endl;
    }
    
    
    std::cout << "#Record(clk_optimization): " << clkList_decision.size()
    << " #Record(clk_contraction):" << clkList_contration.size() << std::endl;
    
    int nrPhase = std::min(clkList_decision.size(), clkList_contration.size());
    
    std::ofstream ofs ("time.txt", std::ofstream::out);
    
    //------------------------------------To Plot------------------------------------//
    /*
     int nrphaseToPlot = 40;
     for (int i = 0; i < std::min(nrphaseToPlot, nrPhase); i++) {
     std::cout << (float) clkList_decision[i] / CLOCKS_PER_SEC << " ";
     }
     std::cout << std::endl;
     
     for (int i = 0; i < std::min(nrphaseToPlot, nrPhase); i++) {
     std::cout << (float) clkList_contration[i] / CLOCKS_PER_SEC << " ";
     }
     std::cout << std::endl;
     */
    
    
    //----------------------------------------------------//
    
    float t_decision = 0, t_contraction = 0;
    
    for (int i = 0; i < clkList_decision.size(); i++) {
        
        t_decision += (float) clkList_decision[i] / CLOCKS_PER_SEC;
        if(i<nrPhase) ofs<< (float) clkList_decision[i] / CLOCKS_PER_SEC<<" ";
        else std::cout<<  (float) clkList_decision[i] / CLOCKS_PER_SEC<<" -> "<<std::endl;
        
    }
    
    ofs<<"\n";
    
    for (int i = 0; i < clkList_contration.size(); i++) {
        t_contraction += (float) clkList_contration[i] / CLOCKS_PER_SEC;
        if(i<nrPhase) ofs<< (float) clkList_contration[i] / CLOCKS_PER_SEC<<" ";
        
    }
    
    ofs<<"\n";
    ofs.close();
    
    std::cout<< " Optimization and contraction time  ratio:"
    << (100 * t_decision)/(t_decision + t_contraction) << " " << (100 * t_contraction)/(t_decision+t_contraction) << std::endl;
    
    
    /*
     for (int i = nrPhase; i < clkList_decision.size(); i++) {
     
     std::cout << clkList_decision[i] << " "
     << (float) clkList_decision[i] / CLOCKS_PER_SEC << std::endl;
     }
     */
    
    std::cout << "(graph):      #V  " << dev_community.g.nb_nodes << " #E   " << dev_community.g.nb_links << std::endl;
    std::cout << "(new graph)  #V  " << dev_community.g_next.nb_nodes << " #E  " << dev_community.g_next.nb_links << std::endl;
    /*
     for (int i = 0; i < n_streams; i++) {
     //CHECK(cudaStreamDestroy(streams[i]));
     }
     */
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //free(streams);
    
    return 0;
}


