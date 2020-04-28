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

#ifndef MYUTILITY_H
#define	MYUTILITY_H

#include"commonconstants.h"
#include"openaddressing.h"
#include"hostconstants.h"
#include"stdio.h"


#ifdef RUNONGPU
__device__
#endif
int lock_community(volatile int* mutex, int id);

#ifdef RUNONGPU
__device__
#endif
void unlock_community(volatile int* mutex, int id);

template <typename Tdata>
#ifdef RUNONGPU
__device__
#endif
Tdata findMaxPerWarp(unsigned int laneId, unsigned int nrElements, Tdata* inputData, unsigned int wrpSz);

__global__ void computeMaxDegreeForWarps(int * indices, int *maxDegreePerWarp,
        int *nrUniDegPerWarp, unsigned int communitySize, unsigned int wrpSz);

__global__ void assign_to_random_communities(int* dev_n2c, int nr_nodes);

__global__ void initialize_locks(int* locks, int nr_communities);

__global__ void initialize_in_tot(int community_size, int* indices, unsigned int* links,
        float* weights, float* tot, float *in, int* n2c, int type, int* locks, unsigned int wrpSz, float* wDegs);

#ifdef RUNONGPU
__device__
#endif
float modularity_gain(double bondness_with_neighbor_com, double w_degree_of_node,
        float tot_of_neighbor_comm, double total_weight);


#ifdef RUNONGPU

__device__
#endif

float compute_weight_of_slef_loops_of_a_node(int node, int laneId, int nr_neighors,
        unsigned int* neighbors, float* weights_of_links_to_neighbors, int type);


#ifdef RUNONGPU
__device__
#endif
float comWDegOfNode(int laneId, int nr_neighors,
        float* weights_of_links_to_neighbors, int type, unsigned int wrpSz);


#ifdef RUNONGPU

__device__
#endif

void remove_node_from_com(int laneid, int node, int comm, float dnodecomm,
        float weighted_degree_of_node, float nr_self_loops, float* in, float* tot,
        int*n2c, int* d_locks, float* in_new, float* tot_new, int*n2c_new);

#ifdef RUNONGPU

__device__
#endif
void insert(int laneid, int node, int comm, double dnodecomm, float weighted_degree_of_node,
        float nr_self_loops, float* in, float* tot, int*n2c, int* d_locks, float* in_new,
        float* tot_new, int*n2c_new);

#ifdef RUNONGPU

__device__
#endif

void copy_from_global_to_shared(int laneId, int segment_len, volatile int* dest, int* src, unsigned int wrpSz);

#ifdef RUNONGPU

__device__
#endif
void compute_neighboring_communites(int node, int laneId, int nr_neighbor,
        unsigned int* communities_of_neighbors, double* bondness_to_neigh_communites,
        unsigned int* neighbors, float* weights_of_links_to_neighbors,
        int *n2c, float *in, int* dev_locks,
        float* tot, float weighted_degree_of_node, float total_weight, int *nr_moves,
        float *in_new, float* tot_new, int* n2c_new, float weight_of_self_loops,
        HashItem* shashTable, unsigned int bucketSize);

#ifdef RUNONGPU

__global__
#endif
void neigh_comm(int community_size, int* indices, unsigned int* links,
        float* weights, int *n2c, float *in, float* tot, int type,
        int *n2c_new, float* tot_new, int* movement_record,
        double total_weight, unsigned int bucketSize,
        int* candidateComms, int nrCandidate, int* primes, int nrPrime,
        int* cardinalityOfComms_old, int* cardinalityOfComms_new,
        unsigned int wrpSz, float *wDegs);

__global__
void get_size_of_communities(int* renumber, int* n2c, int* locks, int nr_nodes);

#ifdef RUNONGPU

__device__
#endif
void combine_same_comm_in_shared_mem(volatile int* myNeighborMem,
        volatile float* myWeightMem, volatile float* combinedWeightMem,
        int nr_read);


#ifdef RUNONGPU

__device__
#endif
void cross_out_duplicate_comm_from_shared_mem(volatile int* myNeighborMem,
        volatile float* myWeightMem, volatile float* combinedWeightMem,
        int nr_read, unsigned int* new_links_warp, float* new_weights_warp,
        bool isFirst_32_of_first_Node, int new_nb_comm,
        int* nr_comm_alread_in_global_mem, int vid);


#ifdef RUNONGPU
__device__
#endif
void compute_neighboring_communites_using_Hash(int node, int laneId, int nr_neighbor,
        unsigned int* communities_of_neighbors, double* bondness_to_neigh_communites,
        unsigned int* neighbors, float* weights_of_links_to_neighbors,
        int *n2c, float *in, int* dev_locks,
        float* tot, float weighted_degree_of_node, float total_weight, int *nr_moves,
        float *in_new, float* tot_new, int* n2c_new, float weight_of_self_loops,
        HashItem* shashTable, unsigned int bucketSize);

__global__
void estimate_size_of_neighborhoods(int* super_node_ptrs, int* indices,
        int* comms_nodes, int new_nb_comm, int* approximate_sizes);
__global__
void computeBoundOfNeighoodSize(int* super_node_ptrs, int* indices,
        int* comms_nodes, int new_nb_comm, int* approximate_sizes,
        unsigned int wrpSz);
__global__
void determine_neighbors_of_new_comms(int* super_node_ptrs, float* new_weights,
        unsigned int* new_links, unsigned int* new_member_counts, int* indices,
        float* weights, unsigned int* links, int* comms_nodes, int new_nb_comm,
        int* n2c, int* renumber, int* start_locations, int type);

__global__
void determineNewNeighborhood(int* super_node_ptrs, float* new_weights,
        unsigned int* new_links, unsigned int* new_member_counts, int* indices,
        float* weights, unsigned int* links, int* comms_nodes, int new_nb_comm,
        int* n2c, int* renumber, int* start_locations, int type,
        unsigned int bucketSize, int* candidateComms, int nrCandidateComms,
        unsigned int wrpSz);

__global__
void group_nodes_based_on_new_CID(int* comm_nodes, int* pos_ptr_of_new_comm,
        int* renumber, int* n2c, int nb_nodes);

__global__
void get_size_of_communities_NEW(int* renumber, int* n2c, int nr_nodes, int* indices);

__global__
void get_size_of_communities(int* renumber, int* n2c, int nr_nodes);
__global__
void findNewNeighodByBlock(int* super_node_ptrs, float* newWeights,
        unsigned int* newLinks, unsigned int* nrNeighborsOfNewComms,
        int* indices, float* weights, unsigned int* links, int* comms_nodes,
        int new_nb_comm, int* n2c, int* renumber, int* start_locations,
        int graphType, unsigned int bucketSize, int* candidateComms,
        int nrCandidateComms, HashItem* gblTable, int* glbTblPtrs,
        int* primes, int nrPrime, unsigned int wrpSz);
#ifdef RUNONGPU

__global__
#endif
void lookAtNeigboringComms(int* indices, unsigned int* links, float* weights,
        int *n2c, float *in, float* tot, int type, int *n2c_new, float *in_new,
        float* tot_new, int* movement_record, double total_weight,
        int* candidateComms, int nrCandidateComms, HashItem* gblTable,
        int* glbTblPtrs, int* primes, int nrPrime, unsigned int wrpSz,
        int* cardinalityOfComms_old, int* cardinalityOfComms_new, float *wDegs);

#ifdef RUNONGPU
__global__
#endif
void reduceGraph(int* indices, unsigned int* links, float* weights, int gType,
        int* uniDegvrts, unsigned int nrUniDegVrts, unsigned int mark,
        int* vtsForPostProcessing, int* n2c);

#ifdef RUNONGPU
__global__
#endif
void editEdgeList(int* indices, unsigned int* links, float* weights, int gType,
        int* uniDegvrts, unsigned int nrUniDegVrts, unsigned int mark,
        int* vtsForPostProcessing);

#ifdef RUNONGPU

__global__
#endif
void preComputeWdegs(int *indices, float* weights, float *wDegs, int type, unsigned int nrComms, int WARP_SIZE);
#ifdef RUNONGPU

__global__
#endif
void preComputePrimes(int *primes, int nrPrimes, unsigned int* thresholds, int nrBigBlock, int *selectedPrimes, int WARP_SIZE);

#ifdef RUNONGPU

__global__
#endif
void changeAssignment(int *n2c, int* n2c_new, int* vertices, int nrVertices);

#ifdef RUNONGPU

__global__
#endif

void computeInternals(int *indices, unsigned int *links, float *weights, int *n2c, float *in, unsigned int nrComms, int graphType);


#ifdef RUNONGPU

__global__
#endif
void update(unsigned int nrComm, float* tot, float* tot_new,
        int* n2c, int* n2c_new, int* cardinalityOfComms,
        int* cardinalityOfComms_new);
#endif	/* MYUTILITY_H */

