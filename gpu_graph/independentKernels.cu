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


#include"communityGPU.h"
#include"commonconstants.h"
#include"openaddressing.h"
#include"myutility.h"
#include"stdio.h"

__global__ void update(unsigned int nrComm, float* tot, float* tot_new,
        int* n2c, int* n2c_new, int* cardinalityOfComms,
        int* cardinalityOfComms_new) {

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < nrComm) {
        n2c[tid] = n2c_new[tid];
        tot[tid] = tot_new[tid];
        cardinalityOfComms[tid] = cardinalityOfComms_new[tid];
    }
}

__global__
void group_nodes_based_on_new_CID(int* comm_nodes, int* pos_ptr_of_new_comm,
        int* oldToNewCidMapping, int* n2c, int nb_nodes) {

    int vid = threadIdx.x + blockIdx.x * blockDim.x;

    while (vid < nb_nodes) {

        int old_cid = n2c[vid];

        int new_cid = oldToNewCidMapping[old_cid];
        if (new_cid >= 0) {
            int current_pos_ptr = atomicAdd(&pos_ptr_of_new_comm[new_cid], 1);

            comm_nodes[current_pos_ptr] = vid;
        }
        vid += blockDim.x * gridDim.x;
    }
}

__global__
void get_size_of_communities_NEW(int* renumber, int* n2c, int nr_nodes, int* indices) {

    int vid = threadIdx.x + blockIdx.x * blockDim.x;

    while (vid < nr_nodes) {

        int commId = n2c[vid];

        if (commId < 0)
            printf("\n PROBLEM commId can't be negative \n");

        int szNhd = indices[vid + 1] - indices[vid];

        if (szNhd > 0) {
            atomicAdd(&renumber[commId], 1);
        }

        vid += blockDim.x * gridDim.x;
    }
}

__global__
void get_size_of_communities(int* renumber, int* n2c, int nr_nodes) {

    int vid = threadIdx.x + blockIdx.x * blockDim.x;

    while (vid < nr_nodes) {

        int commId = n2c[vid];

        if (commId < 0)
            printf("\ncommId can't be negative\n");

        atomicAdd(&renumber[commId], 1);
        vid += blockDim.x * gridDim.x;
    }
}

__global__
void assign_to_random_communities(int* dev_n2c, int nr_nodes) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < nr_nodes) {
        dev_n2c[tid] = tid; // tid % 4;
        tid += blockDim.x * gridDim.x;

    }

}

__global__
void initialize_locks(int* locks, int nr_communities) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < nr_communities) {
        locks[tid] = 0; // all locks are free at the beginning 
        tid += blockDim.x * gridDim.x;
    }
}

__global__
void computeBoundOfNeighoodSize(int* super_node_ptrs, int* indices,
        int* comms_nodes, int new_nb_comm, int* approximate_sizes,
        unsigned int wrpSz) {


    unsigned int wId = threadIdx.x / wrpSz;
    unsigned int laneId = threadIdx.x % wrpSz;

    // Global warp ID
    unsigned int globalWid = blockIdx.x * (blockDim.x / wrpSz) + wId;


    int counter = 0;

    // each warp works with one community 
    if (globalWid < new_nb_comm) {

        int start_of_my_comm = super_node_ptrs[globalWid];
        int end_of_my_comm = super_node_ptrs[globalWid + 1];

        counter = (end_of_my_comm - start_of_my_comm) > 0;

        //Can be parallelized 
        for (int i = start_of_my_comm; i < end_of_my_comm; i++) {

            int vid = comms_nodes[i]; // can you cache comms_nodes ? NO
            //if (globalWid == 63 && !laneId)printf("\n vid (***) = %d |Nhood| = %d \n", vid, (indices[vid + 1] - indices[vid]));
            counter += (indices[vid + 1] - indices[vid]); // can you cache indices ? NO
            //if (globalWid == 63 && !laneId)printf("\n vid (***) = %d |Nhood| = %d counter = %d \n", vid, (indices[vid + 1] - indices[vid]), counter);
        }

        if (!laneId)
            approximate_sizes[globalWid] = counter; // save in global memory
    }
}
#ifdef RUNONGPU

__global__
#endif
void reduceGraph(int* indices, unsigned int* links, float* weights, int gType,
        int* uniDegvrts, unsigned int nrUniDegVrts, unsigned int mark,
        int* vtsForPostProcessing, int* n2c) {

    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < nrUniDegVrts) {

        //vertex Id
        int vid = uniDegvrts[tid];

        // position of its neighbor in indices array
        int pos = indices[vid];

        //unique neighbor of uni-degree vertex
        int uniqNbr = links[pos];
        int degOfNbr = indices[uniqNbr + 1] - indices[uniqNbr];

        if ((degOfNbr > 1) || (uniqNbr < vid)) {

            //mark the edge deleted
            //links[pos] = mark;
            //if (gType == WEIGHTED)weights[pos] = -1;

            n2c[vid] = uniqNbr;
            //save the uniqNbr for post-processing
            vtsForPostProcessing[tid] = uniqNbr;
        } else {
            vtsForPostProcessing[tid] = -1; // nothing to post-process
        }

        tid += blockDim.x * gridDim.x;
    }
}

#ifdef RUNONGPU

__global__
#endif
void editEdgeList(int* indices, unsigned int* links, float* weights, int gType,
        int* uniDegvrts, unsigned int nrUniDegVrts, unsigned int mark,
        int* vtsForPostProcessing) {

    unsigned int wid = threadIdx.x / PHY_WRP_SZ;
    unsigned int laneId = threadIdx.x % PHY_WRP_SZ; // id in the warp


    while (wid < nrUniDegVrts) {

        int vid = vtsForPostProcessing[wid];
        int uniDegVtx = uniDegvrts[wid];
        //Go to the neighbor list of vid and search for uni degree vertex

        int startOfNbrs = indices[vid];
        int endOfNbrs = indices[vid + 1];

        for (int j = startOfNbrs + laneId; j <= endOfNbrs; j = j + PHY_WRP_SZ) {
            if (links[j] == uniDegVtx) {
                links[j] = vid;
                //the weight remains same
            }
        }

        wid = wid + (blockDim.x * gridDim.x) / PHY_WRP_SZ;
    }

}

#ifdef RUNONGPU

__global__
#endif
void changeAssignment(int *n2c, int* n2c_new, int* vertices, int nrVertices) {

    for (int i = 0; i < nrVertices; i++)
        n2c[vertices[i]] = n2c_new[vertices[i]];
}

#ifdef RUNONGPU

__global__
#endif

void computeInternals(int *indices, unsigned int *links, float *weights, int *n2c, float *in, unsigned int nrComms, int graphType) {

    unsigned int vid = threadIdx.x / PHY_WRP_SZ;
    unsigned int laneId = threadIdx.x % PHY_WRP_SZ; // id in the warp

    // Global warp ID and pointer in global Memory
    vid = blockIdx.x * (blockDim.x / PHY_WRP_SZ) + vid;
    while (vid < nrComms) {

        unsigned int startNbr = indices[vid];
        unsigned int endNbr = indices[vid + 1];
        for (unsigned int i = startNbr + laneId; i < endNbr; i = i + PHY_WRP_SZ) {
            unsigned int nbr = links[i];
            if (n2c[nbr] == n2c[vid]) {

                float toAdd = 0.0;

                if (graphType == UNWEIGHTED) {
                    toAdd = 1.0;
                } else {
                    toAdd = weights[i];
                }
                atomicAdd(&in[vid], toAdd);
            }
        }
        vid = vid + (blockDim.x * gridDim.x) / PHY_WRP_SZ;
    }
}


