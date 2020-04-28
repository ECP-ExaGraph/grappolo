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

#include <algorithm>
#include <iostream>
#include "communityGPU.h"

#include <functional>
#include"numeric"

Community::Community(const GraphHOST& input_graph, int nb_pass, double min_mod) {


    //Graph
    g.nb_links = input_graph.nb_links;
    g.nb_nodes = input_graph.nb_nodes;
    g.type = UNWEIGHTED;



    //Copy degree array into indices with an extra zero(0) at the beginning
    g.indices = thrust::device_vector<int>(input_graph.nb_nodes + 1, 0);
    thrust::copy(input_graph.degrees.begin(), input_graph.degrees.end(), g.indices.begin() + 1); // 0 at first position

    /********************Gather Graph Statistics***************/
    std::vector< int> vtxDegs;

    vtxDegs.resize(input_graph.degrees.size());

    std::adjacent_difference(input_graph.degrees.begin(), input_graph.degrees.end(), vtxDegs.begin());

    int totNbrs = std::accumulate(vtxDegs.begin(), vtxDegs.end(), 0);
    int maxDeg = *std::max_element(vtxDegs.begin(), vtxDegs.end());

    double sumSquareDiff = 0;
    double avgDeg = (double) totNbrs / g.nb_nodes;

    for (int i = 0; i < vtxDegs.size(); i++) {
        double delta = ((double) vtxDegs[i] - avgDeg);
        sumSquareDiff += delta*delta;
    }

    double standardDeviation = sqrt(sumSquareDiff / input_graph.nb_nodes);

    std::cout << "MaxDeg = " << maxDeg << " AvgDeg = " << avgDeg << " STD = "
            << standardDeviation << " STD2AvgRatio = " << standardDeviation / avgDeg << std::endl;

    std::cout << "totNbrs =" << totNbrs << " #links =" << input_graph.nb_links << std::endl;

    if (input_graph.nb_nodes < 10) {
        std::cout << std::endl;
        for (int i = 0; i < vtxDegs.size(); i++) {
            std::cout << vtxDegs[i] << " ";
        }
        std::cout << std::endl;
    }
    /**********************************/

    //copy all edges
    g.links.resize(g.nb_links);
    g.links = input_graph.links;

    //copy all weights
    g.weights.resize(input_graph.weights.size());
    g.weights = input_graph.weights;

    std::cout << std::endl << "Copied  " << g.weights.size() << " weights" << std::endl;

    g.total_weight = input_graph.total_weight;


    if (input_graph.weights.size() > 0) {
        g.type = WEIGHTED;
        std::cout << " Setting type to WEIGHTED" << std::endl;
    } else {
        std::cout << "Type is already set to UNWEIGHTED" << std::endl;
    }

    //Community
    community_size = g.nb_nodes;
    min_modularity = min_mod;

    std::cout << std::endl << "(Dev Graph) " << " #Nodes: " << g.nb_nodes << "  #Links: " << g.nb_links / 2 << "  Total_Weight: " << g.total_weight / 2 << std::endl;
    std::cout << "community_size: " << community_size << std::endl;
    // seriously !!
}
