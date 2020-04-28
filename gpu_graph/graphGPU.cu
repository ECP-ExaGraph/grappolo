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

#include "graphGPU.h"
#include"thrust/extrema.h"
#include"thrust/reduce.h"
#include"thrust/execution_policy.h"
#include"thrust/fill.h"
#include"thrust/functional.h"

#ifdef RUNONGPU

__device__
#endif

void copy_from_global_to_shared_(int laneId, int segment_len, volatile int* dest,
        int* src, unsigned int WARP_SIZE) {

    for (int i = laneId; i < segment_len; i = i + WARP_SIZE) {
        dest[i] = src[i];
    }

}


#ifdef RUNONGPU

__device__
#endif
void modify_available_array(int laneId, int nr_neighors, unsigned int* neighbors,
        int* colors, bool* available, bool flag, unsigned int WARP_SIZE) {

    for (int i = laneId; i < nr_neighors; i = i + WARP_SIZE) {

        int nbr = neighbors[laneId];
        int nbrColor = colors[nbr]; // read current color,may be atomic read
        if (nbrColor != -1)
            available[nbrColor] = flag;
    }
}

#ifdef RUNONGPU

__device__
#endif
int pickMinimumColor(int laneId, int maxDegree, bool* available, unsigned int WARP_SIZE) {

    int minColor = 2 * maxDegree + 1;
    for (int c = laneId; c <= maxDegree; c = c + WARP_SIZE) {
        // pick the first available color
        if (available[c]) {
            minColor = c;
            break;
        }
    }
    //decide best
    for (int i = WARP_SIZE / 2; i >= 1; i = i / 2) {
        float tempColor = __shfl_xor(minColor, i, WARP_SIZE);

        if (tempColor < minColor) {
            minColor = tempColor;
        }
    }

    return minColor;
}

/*
 *@available, local array for each warp
 *@colors, contains current coloring
 */
__global__ void coloringKernel(int* indices, unsigned int* links, int* colors,
        bool* available, int numVertices, int maxDegree, unsigned int WARP_SIZE) {

    unsigned int wid = threadIdx.x / WARP_SIZE;
    unsigned int laneId = threadIdx.x % WARP_SIZE; // id in the warp

    extern __shared__ int blockMemory[];

    int* myMemory = blockMemory + (CHUNK_PER_WARP + 1 + CHUNK_PER_WARP) * wid;
    //(CHUNK_PER_WARP+1) indices

    // last (CHUNK_PER_WARP) *************** NEED TO DECIDE******************
    volatile int* warpMemory = myMemory;


    // local warp id
    if (!laneId) {
        //printf(" @tid: %d \n", (wid * WARP_SIZE + laneId));
    }



    // Global warp ID
    wid = blockIdx.x * (NR_THREAD_PER_BLOCK / WARP_SIZE) + wid;

    int num_vertex_to_process = numVertices - wid*CHUNK_PER_WARP;


    if ((wid + 1) * CHUNK_PER_WARP <= numVertices) {
        num_vertex_to_process = CHUNK_PER_WARP;
    }



    if (num_vertex_to_process > 0) {
        // copy indices from global memory to shared memory
        copy_from_global_to_shared_(laneId, num_vertex_to_process + 1,
                warpMemory, &indices[wid * CHUNK_PER_WARP], WARP_SIZE);
    }

    int global_ptr = wid * (maxDegree + 1);
    for (int c = laneId; c <= maxDegree; c = c + WARP_SIZE) {
        available[global_ptr + c] = true;
    }

    //process each vertex sequentially
    for (int vid_index_in_warp = 0; vid_index_in_warp < num_vertex_to_process;
            vid_index_in_warp++) {

        int node = wid * CHUNK_PER_WARP + vid_index_in_warp;
        //if (!wid && !laneId)
        //   printf("For node %d: \n ", node);

        unsigned int start_of_neighbors = warpMemory[vid_index_in_warp];
        unsigned int end_of_neighbors = warpMemory[vid_index_in_warp + 1];



        // make neighboring colors unavailable
        modify_available_array(laneId, (end_of_neighbors - start_of_neighbors),
                &links[start_of_neighbors], colors, &available[global_ptr],
                false, WARP_SIZE);

        if (!laneId && !wid) {
            printf("v=%d:", node);
            for (int k = 0; k <= maxDegree; k++) {
                printf("%d  ", available[global_ptr + k]);

            }
            printf("\n");
        }
        int minColor = pickMinimumColor(laneId, maxDegree, &available[global_ptr], WARP_SIZE);

        if (!laneId)
            colors[node] = minColor;

        if (!laneId && !wid) {
            printf("Picking %d\n", minColor);
        }
        //reset available array
        modify_available_array(laneId, (end_of_neighbors - start_of_neighbors),
                &links[start_of_neighbors], colors, &available[global_ptr], true, WARP_SIZE);


        if (!laneId && !wid) {
            for (int k = 0; k <= maxDegree; k++) {
                printf("%d  ", available[global_ptr + k]);

            }
            printf("\n");
        }

    }
}

void GraphGPU::greedyColoring(unsigned int WARP_SIZE) {

    colors.resize(nb_nodes);

    int load_per_blk = CHUNK_PER_WARP * (NR_THREAD_PER_BLOCK / WARP_SIZE);
    int nr_of_block = (nb_nodes + load_per_blk - 1) / load_per_blk;
    int size_of_shared_memory = (2 * CHUNK_PER_WARP + 1)*(NR_THREAD_PER_BLOCK / WARP_SIZE) * sizeof (int);

    //print flag
    int hostPrint = 1;

    //determine max degree
    thrust::device_vector<int> degree_per_node(nb_nodes);
    thrust::transform(thrust::device, indices.begin() + 1, indices.end(), indices.begin(),
            degree_per_node.begin(), thrust::minus<int>());

    if (hostPrint) {
        std::cout << "Degree per Node:" << std::endl;
        //print_vector(degree_per_node);
    }


    int MAX_DEGREE = *thrust::max_element(degree_per_node.begin(), degree_per_node.end());

    if (hostPrint) {

        std::cout << "Max_Degree: " << MAX_DEGREE << std::endl;
    }
    // available array to be used locally by each warp
    thrust::device_vector<bool> available;
    available.resize((MAX_DEGREE + 1) * nr_of_block * (NR_THREAD_PER_BLOCK / WARP_SIZE));

    // Initialize colors
    thrust::fill_n(thrust::device, colors.begin(), colors.size(), -1);


    if (hostPrint) {
        std::cout << std::endl << "Color:" << std::endl;
        thrust::copy(colors.begin(), colors.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
    }

    coloringKernel << < nr_of_block, NR_THREAD_PER_BLOCK, size_of_shared_memory >>>
            (thrust::raw_pointer_cast(indices.data()), thrust::raw_pointer_cast(links.data()),
            thrust::raw_pointer_cast(colors.data()), thrust::raw_pointer_cast(available.data()),
            nb_nodes, MAX_DEGREE, WARP_SIZE);

    cudaDeviceSynchronize();

    if (hostPrint) {
        std::cout << std::endl << "Color:" << std::endl;
        thrust::copy(colors.begin(), colors.end(), std::ostream_iterator<int>(std::cout, " "));
        std::cout << std::endl;
    }
}
