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

#ifndef GRAPHHOST_H
#define	GRAPHHOST_H

//#include"thrust/host_vector.h"
#include"vector"
#include"assert.h"
#include"commonconstants.h"
class GraphHOST {
public:
    unsigned int nb_nodes;
    unsigned long nb_links;
    double total_weight;
    /*
     thrust::host_vector<unsigned long> degrees;
     thrust::host_vector<unsigned int> links;
     thrust::host_vector<float> weights;
     */
    
    std::vector<unsigned long> degrees;
    std::vector<unsigned int> links;
    std::vector<float> weights;
    
    GraphHOST();
    
    GraphHOST(char *filename, char *filename_w, int type);
    
    // return the weighted degree of the node
    inline double weighted_degree(unsigned int node);
    
    // return the number of neighbors (degree) of the node
    inline unsigned int nb_neighbors(unsigned int node);
    
    //inline thrust::pair<thrust::host_vector<unsigned int>::iterator, thrust::host_vector<float>::iterator >neighbors(unsigned int node);
    inline std::pair<std::vector<unsigned int>::iterator, std::vector<float>::iterator >neighbors(unsigned int node);
    
    void display();
    
};

inline unsigned int
GraphHOST::nb_neighbors(unsigned int node) {
    assert(node >= 0 && node < nb_nodes);
    
    if (node == 0)
        return degrees[0];
    else
        return degrees[node] - degrees[node - 1];
}

//inline thrust::pair<thrust::host_vector<unsigned int>::iterator, thrust::host_vector<float>::iterator >

inline std::pair<std::vector<unsigned int>::iterator, std::vector<float>::iterator >
GraphHOST::neighbors(unsigned int node) {
    assert(node >= 0 && node < nb_nodes);
    
    if (node == 0)
        return std::make_pair(links.begin(), weights.begin());
    else if (weights.size() != 0)
        return std::make_pair(links.begin() + degrees[node - 1], weights.begin() + degrees[node - 1]);
    else
        return std::make_pair(links.begin() + degrees[node - 1], weights.begin());
}

inline double
GraphHOST::weighted_degree(unsigned int node) {
    assert(node >= 0 && node < nb_nodes);
    
    if (weights.size() == 0)
        return (double) nb_neighbors(node);
    else {
        //thrust::pair<thrust::host_vector<unsigned int>::iterator, thrust::host_vector<float>::iterator > p = neighbors(node);
        std::pair<std::vector<unsigned int>::iterator, std::vector<float>::iterator> p = neighbors(node);
        double res = 0;
        for (unsigned int i = 0; i < nb_neighbors(node); i++) {
            res += (double) *(p.second + i);
        }
        return res;
    }
}

#endif	/* GRAPHHOST_H */

