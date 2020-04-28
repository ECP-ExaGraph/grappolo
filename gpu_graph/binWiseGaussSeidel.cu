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
#include"hostconstants.h"
#include"fstream"
#include <thrust/gather.h>

double Community::one_levelGaussSeidel(double init_mod, bool isLastRound,
		int minSize, double easyThreshold, bool isGauss, cudaStream_t *streams,
		int nrStreams, cudaEvent_t &start, cudaEvent_t &stop) {

	//NOTE: cudaStream_t *streams was never used 

	std::cout << std::endl << " Inside method for modularity optimization ";

	if (g.type == WEIGHTED) {
		std::cout << "WEIGHTED Graph" << std::endl;
	} else {
		std::cout << "UnWeighted Graph" << std::endl;
	}

	bool hostPrint = false;
	bool verbose = false;

	int sc;
	sc = 0; //std::cin>>sc;

	if (sc > 1)
		hostPrint = true;

	if (sc > 0)
		verbose = true;

	/*
	   if (hostPrint) {
	   print_vector(g.indices, "indices: ");
	   std::cout << std::endl << "|indices|:" << g.indices.size() << std::endl;
	   }
	 */

	bool improvement = false;
	int nb_moves;
	double cur_mod = -1.0, new_mod = -1.0;

	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);

	unsigned int nrIteration = 0;

	cudaEventRecord(start, 0);

	//Compute degree of each node
	thrust::device_vector<int> sizesOfNhoods(g.indices.size() - 1, 0);


	thrust::transform(g.indices.begin() + 1, g.indices.end(),
			g.indices.begin(), sizesOfNhoods.begin(),
			thrust::minus<int >());

	assert(CAPACITY_FACTOR_DENOMINATOR >= CAPACITY_FACTOR_NUMERATOR);

	// Filters for bins
	// (-1) to hash the community id itself

	int warpLimit = (WARP_TABLE_SIZE_1 * CAPACITY_FACTOR_NUMERATOR / CAPACITY_FACTOR_DENOMINATOR) - 1;
	int blkSMemLimit = (SHARED_TABLE_SIZE * CAPACITY_FACTOR_NUMERATOR / CAPACITY_FACTOR_DENOMINATOR) - 1;
	/*

	   std::cout << "warpLimit: " << warpLimit << "  blkSMemLimit: " << blkSMemLimit << std::endl;

	 */

	IsGreaterThanLimit<int, int>filterBlkGMem(blkSMemLimit);
	IsInRange<int, int> filterBlkSMem(warpLimit + 1, blkSMemLimit);
	IsInRange<int, int> filterForWrp(33, warpLimit);

	assert(warpLimit > 32);

	IsInRange<int, int> filter_N_leq32(17, 32);
	IsInRange<int, int> filter_N_leq16(9, 16);
	IsInRange<int, int> filter_N_leq8(5, 8);
	IsInRange<int, int> filter_N_leq4(1, 4);

	IsInRange<int, int> filterForNone(0, 0); // node with no neighbors

	//count #work for each bin
	int nrCforBlkGMem = thrust::count_if(thrust::device, sizesOfNhoods.begin(), sizesOfNhoods.end(), filterBlkGMem);
	int nrCforBlkSMem = thrust::count_if(thrust::device, sizesOfNhoods.begin(), sizesOfNhoods.end(), filterBlkSMem);
	int nrCforWrp = thrust::count_if(thrust::device, sizesOfNhoods.begin(), sizesOfNhoods.end(), filterForWrp);
	int nrC_N_leq32 = thrust::count_if(thrust::device, sizesOfNhoods.begin(), sizesOfNhoods.end(), filter_N_leq32);
	int nrC_N_leq16 = thrust::count_if(thrust::device, sizesOfNhoods.begin(), sizesOfNhoods.end(), filter_N_leq16);
	int nrC_N_leq8 = thrust::count_if(thrust::device, sizesOfNhoods.begin(), sizesOfNhoods.end(), filter_N_leq8);
	int nrC_N_leq4 = thrust::count_if(thrust::device, sizesOfNhoods.begin(), sizesOfNhoods.end(), filter_N_leq4);

	int nrCforNone = thrust::count_if(thrust::device, sizesOfNhoods.begin(), sizesOfNhoods.end(), filterForNone);
	/*    
	      std::cout << "distribution: "<< nrC_N_leq4 <<" : "<< nrC_N_leq8<<" : "<< nrC_N_leq16<<" : "<< 
	      nrC_N_leq32<<" : "<<nrCforWrp<<" : "<<nrCforBlkSMem<<" : "<<nrCforBlkGMem<<std::endl;

	      std::cout << "distribution: "<< (100*nrC_N_leq4)/community_size <<" : "<< (100*nrC_N_leq8)/community_size<<" : "<< (100*nrC_N_leq16)/community_size <<" : "<< 
	      (100*nrC_N_leq32)/community_size<<" : "<<(100*nrCforWrp)/community_size<<" : "<<(100*nrCforBlkSMem)/community_size<<" : "<<(100*nrCforBlkGMem)/community_size <<std::endl;
	 */
	// Just for statistics 
	IsInRange<int, int> filter_N_leq64(33, 64);
	IsInRange<int, int> filter_N_leq96(65, 96);
	int nrC_N_leq64 = thrust::count_if(thrust::device, sizesOfNhoods.begin(), sizesOfNhoods.end(), filter_N_leq64);
	int nrC_N_leq96 = thrust::count_if(thrust::device, sizesOfNhoods.begin(), sizesOfNhoods.end(), filter_N_leq96);


	int maxNrWrp = thrust::max(thrust::max(thrust::max(nrCforWrp, nrC_N_leq32), thrust::max(nrC_N_leq16, nrC_N_leq8)), nrC_N_leq4);
	/*

	   if (1) {
	   std::cout << "-------> nrCforBlk[" << blkSMemLimit + 1 << ", -] : " << nrCforBlkGMem << std::endl;
	   std::cout << "-------> nrCforBlk[" << warpLimit + 1 << "," << blkSMemLimit << "] : " << nrCforBlkSMem << std::endl;
	   std::cout << "----------> nrCforWrp[ 33, " << warpLimit << "] : " << nrCforWrp << std::endl;

	   std::cout << "nrC_N_leq32 :" << nrC_N_leq32 << std::endl;
	   std::cout << "nrC_N_leq16 :" << nrC_N_leq16 << std::endl;
	   std::cout << "nrC_N_leq8 :" << nrC_N_leq8 << std::endl;

	   std::cout << "nrC_N_leq4 :" << nrC_N_leq4 << std::endl;

	   std::cout << "----------> nrCforNone[0,0] : " << nrCforNone << std::endl;

	   std::cout << "maxNrWrp :" << maxNrWrp << std::endl;

	   std::cout << "----------Statistics----------------" << std::endl;
	   std::cout << "nrC_N_leq64 :" << nrC_N_leq64 << std::endl;
	   std::cout << "nrC_N_leq96 :" << nrC_N_leq96 << std::endl;
	   }
	 */

	assert((nrCforBlkGMem + nrCforBlkSMem + nrCforWrp + nrC_N_leq32 + nrC_N_leq16 + nrC_N_leq8 + nrC_N_leq4 + nrCforNone) == community_size);

	thrust::device_vector<int> movement_counters(maxNrWrp, 0);

	//Lets copy Identities of all communities  in g_next.links

	g_next.links.resize(community_size, 0);
	thrust::sequence(g_next.links.begin(), g_next.links.end(), 0);


	//Use g_next.indices to copy community ids with decreasing sizes of neighborhood

	g_next.indices.resize(community_size, -1);

	//First community ids with larger neighborhoods
	thrust::copy_if(thrust::device, g_next.links.begin(), g_next.links.end(),
			sizesOfNhoods.begin(), g_next.indices.begin(), filterBlkGMem);

	// Then community ids with medium sized neighborhoods

	thrust::copy_if(thrust::device, g_next.links.begin(), g_next.links.end(), sizesOfNhoods.begin(),
			g_next.indices.begin() + nrCforBlkGMem, filterBlkSMem);

	// Community ids with smaller neighborhoods
	thrust::copy_if(thrust::device, g_next.links.begin(), g_next.links.end(), sizesOfNhoods.begin(),
			g_next.indices.begin() + nrCforBlkGMem + nrCforBlkSMem, filterForWrp);

	thrust::copy_if(thrust::device, g_next.links.begin(), g_next.links.end(), sizesOfNhoods.begin(),
			g_next.indices.begin() + nrCforBlkGMem + nrCforBlkSMem + nrCforWrp, filter_N_leq32);

	thrust::copy_if(thrust::device, g_next.links.begin(), g_next.links.end(), sizesOfNhoods.begin(),
			g_next.indices.begin() + nrCforBlkGMem + nrCforBlkSMem + nrCforWrp + nrC_N_leq32, filter_N_leq16);


	thrust::copy_if(thrust::device, g_next.links.begin(), g_next.links.end(), sizesOfNhoods.begin(),
			g_next.indices.begin() + nrCforBlkGMem + nrCforBlkSMem + nrCforWrp + nrC_N_leq32 + nrC_N_leq16, filter_N_leq8);

	thrust::copy_if(thrust::device, g_next.links.begin(), g_next.links.end(), sizesOfNhoods.begin(),
			g_next.indices.begin() + nrCforBlkGMem + nrCforBlkSMem + nrCforWrp + nrC_N_leq32 + nrC_N_leq16 + nrC_N_leq8, filter_N_leq4);
	///////////////////////////////////////////////////

	// Now, use g_next.links to copy sizes of neighborhood according to order given by g_next.indices

	g_next.links.resize(g_next.indices.size(), 0);

	thrust::gather(thrust::device, g_next.indices.begin(), g_next.indices.end(), sizesOfNhoods.begin(), g_next.links.begin());

	//Sort according to size of neighborhood ; only first nrCforBlkGbMem

	thrust::sort_by_key(g_next.links.begin(), g_next.links.begin() + nrCforBlkGMem,
			g_next.indices.begin(), thrust::greater<unsigned int>());

	//////////Just to debug /////////////////
	/*
	   if (0) {
	   thrust::host_vector<int> esSizes = sizesOfNhoods;
	   thrust::host_vector<int> bigCommunites = g_next.indices;
	   for (int k = 0; k < thrust::min<int>(bigCommunites.size(), 5); k++) {
	   std::cout << bigCommunites[k] << "::" << esSizes[bigCommunites[k]] << std::endl;
	   }
	   esSizes.clear();
	   bigCommunites.clear();
	   }

	 */
	///////////////////////////////Allocate data for Global HashTable////////////////////

	int nrBlockForLargeNhoods = 90;
	nrBlockForLargeNhoods = thrust::min(thrust::max(nrCforBlkGMem, nrCforBlkSMem), nrBlockForLargeNhoods);

	thrust::device_vector<int> hashTablePtrs(nrBlockForLargeNhoods + 1, 0);

	//g_next.links contains sizes of big neighborhoods

	thrust::inclusive_scan(g_next.links.begin(), g_next.links.begin() + nrBlockForLargeNhoods,
			hashTablePtrs.begin() + 1, thrust::plus<int>());


	thrust::device_vector<HashItem> globalHashTable(2 * hashTablePtrs.back());

	int szHTmem = thrust::reduce(g_next.links.begin(), g_next.links.begin() + nrBlockForLargeNhoods, (int) 0);

	//std::cout << globalHashTable.size() << ":" << 2 * szHTmem << std::endl;

	thrust::device_vector<int> moveCounters(nrBlockForLargeNhoods, 0);




	////////////////////////////////////////////////////



	unsigned int wrpSz = PHY_WRP_SZ;
	int nr_of_block = 0;


	//std::cout << " g.weight:(copied from host graph) " << g.total_weight << std::endl;

	//////////////////////////////////////////////////////////////

	n2c.resize(community_size);

	thrust::sequence(n2c.begin(), n2c.end(), 0);

	//std::cout << "community_size : " << community_size << " n2c.size : " << n2c.size() << std::endl;

	thrust::device_vector< int> n2c_old(n2c.size(), -1);

	assert(community_size == n2c.size());


	g.total_weight = 0.0;

	if (g.type == WEIGHTED) {
		g.total_weight = thrust::reduce(thrust::device, g.weights.begin(), g.weights.end(), (double) 0, thrust::plus<double>());
	} else {
		g.total_weight = (double) g.nb_links;
	}

	report_time(start, stop, "FilterCopy&M");
	//std::cout << " g.weight(computed in device): " << g.total_weight << std::endl;

	thrust::device_vector< int> cardinalityOfComms(community_size, 1); // cardinality of each community

	thrust::device_vector< int> cardinalityOfComms_new(community_size, 0); // cardinality of each community

	thrust::device_vector<float> tot_new(community_size, 0.0);

	thrust::device_vector<float> tot(community_size, 0.0);
	thrust::device_vector<float> in(community_size, 0.0);

	in.resize(community_size);
	tot.resize(community_size);
	tot_new.resize(community_size);

	// n2c_new.clear();
	n2c_new.resize(community_size);

	/////////////////////////////////////////////////////////////

	wrpSz = PHY_WRP_SZ;
	int load_per_blk = CHUNK_PER_WARP * (NR_THREAD_PER_BLOCK / wrpSz);
	nr_of_block = (community_size + load_per_blk - 1) / load_per_blk;

	thrust::device_vector<float> wDegs(community_size, 0.0);

	cudaEventRecord(start, 0);
	preComputeWdegs << <nr_of_block, NR_THREAD_PER_BLOCK>>>(thrust::raw_pointer_cast(g.indices.data()),
			thrust::raw_pointer_cast(g.weights.data()),
			thrust::raw_pointer_cast(wDegs.data()),
			g.type, community_size, wrpSz);

	report_time(start, stop, "preComputeWdegs");
	//////////////////////////////////////////////////////////////	

	wrpSz = PHY_WRP_SZ;
	load_per_blk = CHUNK_PER_WARP * (NR_THREAD_PER_BLOCK / wrpSz);
	nr_of_block = (community_size + load_per_blk - 1) / load_per_blk;
	int size_of_shared_memory = (2 * CHUNK_PER_WARP + 1)*(NR_THREAD_PER_BLOCK / wrpSz) * sizeof (int);


	cudaEventRecord(start, 0);

	initialize_in_tot << < nr_of_block, NR_THREAD_PER_BLOCK, size_of_shared_memory >>>(community_size,
			thrust::raw_pointer_cast(g.indices.data()), thrust::raw_pointer_cast(g.links.data()),
			thrust::raw_pointer_cast(g.weights.data()), thrust::raw_pointer_cast(tot.data()),
			NULL, thrust::raw_pointer_cast(n2c.data()), g.type, NULL, wrpSz,
			thrust::raw_pointer_cast(wDegs.data()));

	report_time(start, stop, "initialize_in_tot");

	//////////////////////////////////////

	int loopCnt = 0;


	double threshold = min_modularity;
	if (community_size > minSize && isLastRound == false)
		threshold = easyThreshold;

	std::cout<<"Status::  community size - "<<community_size<<" threshold - "<<threshold<<std::endl;
	//  std::cout << "minSize: " << minSize << std::endl;


	//NEVER set it to TRUE; it doesn't work!!!!!!!!!!!
	bool isToUpdate = false; // true;

	clock_t t1, t2;
	do {
		t1 = clock();

		loopCnt++;
		//   std::cout << " ---------------------------- do-while ---------------------" << loopCnt << std::endl;


		thrust::fill_n(thrust::device, in.begin(), in.size(), 0.0); // initialize in to all zeros '0'
		n2c_new = n2c; // MUST NEEDED Assignment
		tot_new = tot;
		cardinalityOfComms_new = cardinalityOfComms;



		//thrust::fill_n(thrust::device, tot_new.begin(), tot_new.size(),0.0);
		//thrust::fill_n(thrust::device, cardinalityOfComms_new.begin(), cardinalityOfComms_new.size(),0);

		nb_moves = 0;
		unsigned int bucketSizePerWarp = 0;
		size_t sizeHashMem = 0;

		// Initialize counters 
		movement_counters.clear();
		movement_counters.resize(maxNrWrp, 0);

		moveCounters.clear();
		moveCounters.resize(nrBlockForLargeNhoods, 0);
		if (nrCforBlkGMem > 0) {

			if (isGauss) {
				//thrust::fill_n(thrust::device, in.begin(), in.size(), 0.0); // initialize in to all zeros '0'
				//tot_new = tot;
				//cardinalityOfComms_new = cardinalityOfComms;
			}

			wrpSz = PHY_WRP_SZ;

			cudaEventRecord(start, 0);

			//std::cout<<" nrBlockForLargeNhoods: "<<nrBlockForLargeNhoods<<" nrCforBlkGMem:  "<<  nrCforBlkGMem<<std::endl;
			lookAtNeigboringComms << <nrBlockForLargeNhoods, (NR_THREAD_PER_BLOCK * 2)>>>(
					thrust::raw_pointer_cast(g.indices.data()),
					thrust::raw_pointer_cast(g.links.data()),
					thrust::raw_pointer_cast(g.weights.data()),
					thrust::raw_pointer_cast(n2c.data()),
					thrust::raw_pointer_cast(in.data()),
					thrust::raw_pointer_cast(tot.data()), g.type,
					thrust::raw_pointer_cast(n2c_new.data()),
					NULL,
					thrust::raw_pointer_cast(tot_new.data()),
					thrust::raw_pointer_cast(moveCounters.data()), g.total_weight,
					thrust::raw_pointer_cast(g_next.indices.data()), nrCforBlkGMem,
					thrust::raw_pointer_cast(globalHashTable.data()),
					thrust::raw_pointer_cast(hashTablePtrs.data()),
					thrust::raw_pointer_cast(devPrimes.data()), nb_prime, wrpSz,
					thrust::raw_pointer_cast(cardinalityOfComms.data()),
					thrust::raw_pointer_cast(cardinalityOfComms_new.data()),
					thrust::raw_pointer_cast(wDegs.data()));

			report_time(start, stop, "lookAtNeigboringComms");
			//nb_moves = nb_moves + thrust::reduce(moveCounters.begin(), moveCounters.begin() + nrBlockForLargeNhoods, (int) 0);

			/*
			   if (0) {
			   changeAssignment << < nr_of_block, NR_THREAD_PER_BLOCK>>>(
			   thrust::raw_pointer_cast(n2c.data()), // change from
			   thrust::raw_pointer_cast(n2c_new.data()), // change to
			   thrust::raw_pointer_cast(g_next.indices.data()), nrCforBlkGMem);
			   }
			 */

			if (isGauss) {
				if (isToUpdate) {
					nr_of_block = (community_size + NR_THREAD_PER_BLOCK - 1) / NR_THREAD_PER_BLOCK;
					update << <nr_of_block, NR_THREAD_PER_BLOCK>>>(community_size,
							thrust::raw_pointer_cast(tot.data()),
							thrust::raw_pointer_cast(tot_new.data()),
							thrust::raw_pointer_cast(n2c.data()),
							thrust::raw_pointer_cast(n2c_new.data()),
							thrust::raw_pointer_cast(cardinalityOfComms.data()),
							thrust::raw_pointer_cast(cardinalityOfComms_new.data()));
				} else {
					n2c = n2c_new;
					tot = tot_new;
					cardinalityOfComms = cardinalityOfComms_new;
				}
			}
		}

		/*

		   if (verbose) {
		   std::cout << "Before Traversing.............. " << std::endl;
		   print_vector(tot, "tot:");
		   }

		   if (verbose) {
		   std::cout << " community_size:" << community_size << std::endl;
		   std::cout << " g.indices:" << g.indices.size() << std::endl;
		   std::cout << " g.links:" << g.links.size() << std::endl;
		   std::cout << " g.weights:" << g.weights.size() << std::endl;
		   std::cout << " n2c:" << n2c.size() << std::endl;
		   std::cout << " in:" << in.size() << std::endl;

		   std::cout << " n2c_new:" << n2c_new.size() << std::endl;
		   std::cout << " tot_new:" << tot_new.size() << std::endl;
		   std::cout << " movement_counters: " << movement_counters.size() << std::endl;
		   std::cout << " g.total_weight: " << g.total_weight << std::endl;

		   }

		   if (verbose) {
		   nb_moves = thrust::reduce(movement_counters.begin(), movement_counters.end(), (int) 0);
		   std::cout << "---------*Now*  " << nb_moves << std::endl;
		   }

		 */
		sc = 0; //std::cin>>sc;

		//////////////////////////////////////////////////

		/*
		   if (nrC_N_leq32) {

		   if (isGauss) {
		   thrust::fill_n(thrust::device, in.begin(), in.size(), 0.0); // initialize in to all zeros '0'
		//tot_new = tot;
		//cardinalityOfComms_new = cardinalityOfComms;
		}

		wrpSz = PHY_WRP_SZ;
		nr_of_block = (nrC_N_leq32 + (NR_THREAD_PER_BLOCK / wrpSz) - 1) / (NR_THREAD_PER_BLOCK / wrpSz);

		bucketSizePerWarp = 61; //MUST BE PRIME
		sizeHashMem = (NR_THREAD_PER_BLOCK / wrpSz) * bucketSizePerWarp * sizeof (HashItem);

		cudaEventRecord(start, 0);

		neigh_comm << < nr_of_block, NR_THREAD_PER_BLOCK, sizeHashMem >>>(
		community_size,
		thrust::raw_pointer_cast(g.indices.data()),
		thrust::raw_pointer_cast(g.links.data()),
		thrust::raw_pointer_cast(g.weights.data()),
		thrust::raw_pointer_cast(n2c.data()),
		thrust::raw_pointer_cast(in.data()),
		thrust::raw_pointer_cast(tot.data()), g.type,
		thrust::raw_pointer_cast(n2c_new.data()),
		thrust::raw_pointer_cast(tot_new.data()),
		thrust::raw_pointer_cast(movement_counters.data()),
		g.total_weight, bucketSizePerWarp,
		thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem + nrCforBlkSMem + nrCforWrp,
		nrC_N_leq32, thrust::raw_pointer_cast(devPrimes.data()), nb_prime,
		thrust::raw_pointer_cast(cardinalityOfComms.data()),
		thrust::raw_pointer_cast(cardinalityOfComms_new.data()),
		wrpSz, thrust::raw_pointer_cast(wDegs.data()));

		report_time(start, stop, "neigh_comm(<=32)");
		nb_moves = nb_moves + thrust::reduce(movement_counters.begin(), movement_counters.begin() + nrC_N_leq32, (int) 0);

		if(0){
		changeAssignment<<< nr_of_block, NR_THREAD_PER_BLOCK>>>(  
		thrust::raw_pointer_cast(n2c.data()), // change from
		thrust::raw_pointer_cast(n2c_new.data()), // change to
		thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem + nrCforBlkSMem + nrCforWrp,
		nrC_N_leq32);
		}

		if (isGauss) {
		n2c = n2c_new;
		tot = tot_new;
		cardinalityOfComms = cardinalityOfComms_new;
		}
		}




		if (nrCforWrp) {

		if (isGauss) {
		thrust::fill_n(thrust::device, in.begin(), in.size(), 0.0); // initialize in to all zeros '0'
		//tot_new = tot;
		//cardinalityOfComms_new = cardinalityOfComms;
		}


		wrpSz = PHY_WRP_SZ;
		nr_of_block = (nrCforWrp + (NR_THREAD_PER_BLOCK / wrpSz) - 1) / (NR_THREAD_PER_BLOCK / wrpSz);

		bucketSizePerWarp = WARP_TABLE_SIZE_1;
		sizeHashMem = (NR_THREAD_PER_BLOCK / wrpSz) * bucketSizePerWarp * sizeof (HashItem);

		cudaEventRecord(start, 0);

		neigh_comm << < nr_of_block, NR_THREAD_PER_BLOCK, sizeHashMem >>>(
				community_size,
				thrust::raw_pointer_cast(g.indices.data()),
				thrust::raw_pointer_cast(g.links.data()),
				thrust::raw_pointer_cast(g.weights.data()),
				thrust::raw_pointer_cast(n2c.data()),
				thrust::raw_pointer_cast(in.data()),
				thrust::raw_pointer_cast(tot.data()), g.type,
				thrust::raw_pointer_cast(n2c_new.data()),
				thrust::raw_pointer_cast(tot_new.data()),
				thrust::raw_pointer_cast(movement_counters.data()),
				g.total_weight, bucketSizePerWarp,
				thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem + nrCforBlkSMem,
				nrCforWrp, thrust::raw_pointer_cast(devPrimes.data()), nb_prime,
				thrust::raw_pointer_cast(cardinalityOfComms.data()),
				thrust::raw_pointer_cast(cardinalityOfComms_new.data()),
				wrpSz, thrust::raw_pointer_cast(wDegs.data()));

		report_time(start, stop, "neigh_comm");
		nb_moves = thrust::reduce(movement_counters.begin(), movement_counters.begin() + nrCforWrp, (int) 0);

		// change community assignment of processed vertices
		if (0) {
			changeAssignment << < nr_of_block, NR_THREAD_PER_BLOCK>>>(
					thrust::raw_pointer_cast(n2c.data()), // change from
					thrust::raw_pointer_cast(n2c_new.data()), // change to
					thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem + nrCforBlkSMem, // of these communities
					nrCforWrp);
		}

		if (isGauss) {
			n2c = n2c_new;
			tot = tot_new;
			cardinalityOfComms = cardinalityOfComms_new;
		}
	}






	if (nrCforBlkGMem > 0) {

		if (isGauss) {
			thrust::fill_n(thrust::device, in.begin(), in.size(), 0.0); // initialize in to all zeros '0'
			//tot_new = tot;
			//cardinalityOfComms_new = cardinalityOfComms;
		}

		wrpSz = PHY_WRP_SZ;

		cudaEventRecord(start, 0);

		lookAtNeigboringComms << <nrBlockForLargeNhoods, (NR_THREAD_PER_BLOCK * 2)>>>(
				thrust::raw_pointer_cast(g.indices.data()),
				thrust::raw_pointer_cast(g.links.data()),
				thrust::raw_pointer_cast(g.weights.data()),
				thrust::raw_pointer_cast(n2c.data()),
				thrust::raw_pointer_cast(in.data()),
				thrust::raw_pointer_cast(tot.data()), g.type,
				thrust::raw_pointer_cast(n2c_new.data()),
				NULL,
				thrust::raw_pointer_cast(tot_new.data()),
				thrust::raw_pointer_cast(moveCounters.data()), g.total_weight,
				thrust::raw_pointer_cast(g_next.indices.data()), nrCforBlkGMem,
				thrust::raw_pointer_cast(globalHashTable.data()),
				thrust::raw_pointer_cast(hashTablePtrs.data()),
				thrust::raw_pointer_cast(devPrimes.data()), nb_prime, wrpSz,
				thrust::raw_pointer_cast(cardinalityOfComms.data()),
				thrust::raw_pointer_cast(cardinalityOfComms_new.data()),
				thrust::raw_pointer_cast(wDegs.data()));

		report_time(start, stop, "lookAtNeigboringComms");
		nb_moves = nb_moves + thrust::reduce(moveCounters.begin(), moveCounters.begin() + nrBlockForLargeNhoods, (int) 0);

		if (0) {
			changeAssignment << < nr_of_block, NR_THREAD_PER_BLOCK>>>(
					thrust::raw_pointer_cast(n2c.data()), // change from
					thrust::raw_pointer_cast(n2c_new.data()), // change to
					thrust::raw_pointer_cast(g_next.indices.data()), nrCforBlkGMem);
		}

		if (isGauss) {
			n2c = n2c_new;
			tot = tot_new;
			cardinalityOfComms = cardinalityOfComms_new;
		}
	}

	if (nrCforBlkSMem > 0) {


		if (isGauss) {
			thrust::fill_n(thrust::device, in.begin(), in.size(), 0.0); // initialize in to all zeros '0'
			//tot_new = tot;
			//cardinalityOfComms_new = cardinalityOfComms;
		}

		wrpSz = PHY_WRP_SZ;
		cudaEventRecord(start, 0);

		lookAtNeigboringComms << <nrBlockForLargeNhoods, NR_THREAD_PER_BLOCK>>>(
				thrust::raw_pointer_cast(g.indices.data()),
				thrust::raw_pointer_cast(g.links.data()),
				thrust::raw_pointer_cast(g.weights.data()),
				thrust::raw_pointer_cast(n2c.data()),
				thrust::raw_pointer_cast(in.data()),
				thrust::raw_pointer_cast(tot.data()), g.type,
				thrust::raw_pointer_cast(n2c_new.data()),
				NULL,
				thrust::raw_pointer_cast(tot_new.data()),
				thrust::raw_pointer_cast(moveCounters.data()), g.total_weight,
				thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem, nrCforBlkSMem,
				thrust::raw_pointer_cast(globalHashTable.data()),
				thrust::raw_pointer_cast(hashTablePtrs.data()),
				thrust::raw_pointer_cast(devPrimes.data()), nb_prime, wrpSz,
				thrust::raw_pointer_cast(cardinalityOfComms.data()),
				thrust::raw_pointer_cast(cardinalityOfComms_new.data()),
				thrust::raw_pointer_cast(wDegs.data()));
		report_time(start, stop, "lookAtNeigboringComms(sh)");
		nb_moves = nb_moves + thrust::reduce(moveCounters.begin(), moveCounters.begin() + nrBlockForLargeNhoods, (int) 0);

		if (0) {
			changeAssignment << < nr_of_block, NR_THREAD_PER_BLOCK>>>(
					thrust::raw_pointer_cast(n2c.data()), // change from
					thrust::raw_pointer_cast(n2c_new.data()), // change to
					thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem, nrCforBlkSMem);
		}

		if (isGauss) {
			n2c = n2c_new;
			tot = tot_new;
			cardinalityOfComms = cardinalityOfComms_new;
		}
	}
	*/
		//////////////////////////////////////////////////


		if (nrC_N_leq8) {


			if (isGauss) {
				//thrust::fill_n(thrust::device, in.begin(), in.size(), 0.0); // initialize in to all zeros '0'
				//tot_new = tot;
				//cardinalityOfComms_new = cardinalityOfComms;
			}
			wrpSz = QUARTER_WARP;
			nr_of_block = (nrC_N_leq8 + (NR_THREAD_PER_BLOCK / wrpSz) - 1) / (NR_THREAD_PER_BLOCK / wrpSz);

			bucketSizePerWarp = 17; // MUST BE PRIME
			sizeHashMem = (NR_THREAD_PER_BLOCK / wrpSz) * bucketSizePerWarp * sizeof (HashItem);

			/*
			   if (0) {
			   std::cin>>sc;

			   print_vector(g.indices, "g.indices: ");
			   print_vector(g.links, "g.links: ");
			   print_vector(n2c, "n2c:");
			   print_vector(in, "in: ");
			   print_vector(tot, "tot:");
			   print_vector(n2c_new, "n2c_new:");
			   print_vector(tot_new, "tot_new:");
			   print_vector(movement_counters, "movement_counters:");
			   print_vector(g_next.indices, "g_next.indices:");
			   print_vector(devPrimes, "devPrimes:");
			   print_vector(cardinalityOfComms, "cardinalityOfComms:");
			   print_vector(cardinalityOfComms_new, "cardinalityOfComms_new:");
			   }
			 */
			cudaEventRecord(start, 0);

			//print_vector(in, "in (*): ");
			neigh_comm << < nr_of_block, NR_THREAD_PER_BLOCK, sizeHashMem >>>(
					community_size,
					thrust::raw_pointer_cast(g.indices.data()),
					thrust::raw_pointer_cast(g.links.data()),
					thrust::raw_pointer_cast(g.weights.data()),
					thrust::raw_pointer_cast(n2c.data()),
					thrust::raw_pointer_cast(in.data()),
					thrust::raw_pointer_cast(tot.data()), g.type,
					thrust::raw_pointer_cast(n2c_new.data()),
					thrust::raw_pointer_cast(tot_new.data()),
					thrust::raw_pointer_cast(movement_counters.data()),
					g.total_weight, bucketSizePerWarp,
					thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem + nrCforBlkSMem + nrCforWrp + nrC_N_leq32 + nrC_N_leq16,
					nrC_N_leq8, thrust::raw_pointer_cast(devPrimes.data()), nb_prime,
					thrust::raw_pointer_cast(cardinalityOfComms.data()),
					thrust::raw_pointer_cast(cardinalityOfComms_new.data()),
					wrpSz, thrust::raw_pointer_cast(wDegs.data()));

			//print_vector(in, "in (*): ");
			report_time(start, stop, "neigh_comm ( <=8)");
			//nb_moves = nb_moves + thrust::reduce(movement_counters.begin(), movement_counters.begin() + nrC_N_leq8, (int) 0);

			/*
			   changeAssignment<<< nr_of_block, NR_THREAD_PER_BLOCK>>>(  
			   thrust::raw_pointer_cast(n2c.data()), // change from
			   thrust::raw_pointer_cast(n2c_new.data()), // change to
			   thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem + nrCforBlkSMem + nrCforWrp + nrC_N_leq32 + nrC_N_leq16,
			   nrC_N_leq8);
			 */

			if (isGauss) {

				if (isToUpdate) {
					assert(community_size == n2c.size());
					assert(community_size == n2c_new.size());
					assert(community_size == tot.size());
					assert(community_size == tot_new.size());
					assert(community_size == cardinalityOfComms.size());
					assert(community_size == cardinalityOfComms_new.size());


					nr_of_block = (community_size + NR_THREAD_PER_BLOCK - 1) / NR_THREAD_PER_BLOCK;
					update << <nr_of_block, NR_THREAD_PER_BLOCK>>>(community_size,
							thrust::raw_pointer_cast(tot.data()),
							thrust::raw_pointer_cast(tot_new.data()),
							thrust::raw_pointer_cast(n2c.data()),
							thrust::raw_pointer_cast(n2c_new.data()),
							thrust::raw_pointer_cast(cardinalityOfComms.data()),
							thrust::raw_pointer_cast(cardinalityOfComms_new.data()));
				} else {
					n2c = n2c_new;
					tot = tot_new;
					cardinalityOfComms = cardinalityOfComms_new;
				}
			}
		}

	if (nrC_N_leq16) {

		if (isGauss) {
			//thrust::fill_n(thrust::device, in.begin(), in.size(), 0.0); // initialize in to all zeros '0'
			//tot_new = tot;
			//cardinalityOfComms_new = cardinalityOfComms;
		}

		wrpSz = HALF_WARP;
		nr_of_block = (nrC_N_leq16 + (NR_THREAD_PER_BLOCK / wrpSz) - 1) / (NR_THREAD_PER_BLOCK / wrpSz);

		bucketSizePerWarp = 31; // MUST BE PRIME
		sizeHashMem = (NR_THREAD_PER_BLOCK / wrpSz) * bucketSizePerWarp * sizeof (HashItem);

		cudaEventRecord(start, 0);

		neigh_comm << < nr_of_block, NR_THREAD_PER_BLOCK, sizeHashMem >>>(
				community_size,
				thrust::raw_pointer_cast(g.indices.data()),
				thrust::raw_pointer_cast(g.links.data()),
				thrust::raw_pointer_cast(g.weights.data()),
				thrust::raw_pointer_cast(n2c.data()),
				thrust::raw_pointer_cast(in.data()),
				thrust::raw_pointer_cast(tot.data()), g.type,
				thrust::raw_pointer_cast(n2c_new.data()),
				thrust::raw_pointer_cast(tot_new.data()),
				thrust::raw_pointer_cast(movement_counters.data()),
				g.total_weight, bucketSizePerWarp,
				thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem + nrCforBlkSMem + nrCforWrp + nrC_N_leq32,
				nrC_N_leq16, thrust::raw_pointer_cast(devPrimes.data()), nb_prime,
				thrust::raw_pointer_cast(cardinalityOfComms.data()),
				thrust::raw_pointer_cast(cardinalityOfComms_new.data()),
				wrpSz, thrust::raw_pointer_cast(wDegs.data()));

		report_time(start, stop, "neigh_comm ( <=16)");
		//nb_moves = nb_moves + thrust::reduce(movement_counters.begin(), movement_counters.begin() + nrC_N_leq16, (int) 0);

		/*
		   changeAssignment<<< nr_of_block, NR_THREAD_PER_BLOCK>>>(  
		   thrust::raw_pointer_cast(n2c.data()), // change from
		   thrust::raw_pointer_cast(n2c_new.data()), // change to
		   thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem + nrCforBlkSMem + nrCforWrp + nrC_N_leq32,
		   nrC_N_leq16);
		 */

		if (isGauss) {
			if (isToUpdate) {
				nr_of_block = (community_size + NR_THREAD_PER_BLOCK - 1) / NR_THREAD_PER_BLOCK;
				update << <nr_of_block, NR_THREAD_PER_BLOCK>>>(community_size,
						thrust::raw_pointer_cast(tot.data()),
						thrust::raw_pointer_cast(tot_new.data()),
						thrust::raw_pointer_cast(n2c.data()),
						thrust::raw_pointer_cast(n2c_new.data()),
						thrust::raw_pointer_cast(cardinalityOfComms.data()),
						thrust::raw_pointer_cast(cardinalityOfComms_new.data()));
			} else {
				n2c = n2c_new;
				tot = tot_new;
				cardinalityOfComms = cardinalityOfComms_new;
			}
		}
	}




	if (nrC_N_leq4) {


		if (isGauss) {
			//thrust::fill_n(thrust::device, in.begin(), in.size(), 0.0); // initialize in to all zeros '0'
			//tot_new = tot;
			//cardinalityOfComms_new = cardinalityOfComms;
		}
		wrpSz = QUARTER_WARP / 2;
		nr_of_block = (nrC_N_leq4 + (NR_THREAD_PER_BLOCK / wrpSz) - 1) / (NR_THREAD_PER_BLOCK / wrpSz);

		bucketSizePerWarp = 7; // MUST BE PRIME
		sizeHashMem = (NR_THREAD_PER_BLOCK / wrpSz) * bucketSizePerWarp * sizeof (HashItem);
		/*

		   if (0) {
		   std::cin>>sc;

		   print_vector(g.indices, "g.indices: ");
		   print_vector(g.links, "g.links: ");
		   print_vector(n2c, "n2c:");
		   print_vector(in, "in: ");
		   print_vector(tot, "tot:");
		   print_vector(n2c_new, "n2c_new:");
		   print_vector(tot_new, "tot_new:");
		   print_vector(movement_counters, "movement_counters:");
		   print_vector(g_next.indices, "g_next.indices:");
		   print_vector(devPrimes, "devPrimes:");
		   print_vector(cardinalityOfComms, "cardinalityOfComms:");
		   print_vector(cardinalityOfComms_new, "cardinalityOfComms_new:");
		   }
		 */
		cudaEventRecord(start, 0);

		//print_vector(in, "in (*): ");
		neigh_comm << < nr_of_block, NR_THREAD_PER_BLOCK, sizeHashMem >>>(
				community_size,
				thrust::raw_pointer_cast(g.indices.data()),
				thrust::raw_pointer_cast(g.links.data()),
				thrust::raw_pointer_cast(g.weights.data()),
				thrust::raw_pointer_cast(n2c.data()),
				thrust::raw_pointer_cast(in.data()),
				thrust::raw_pointer_cast(tot.data()), g.type,
				thrust::raw_pointer_cast(n2c_new.data()),
				thrust::raw_pointer_cast(tot_new.data()),
				thrust::raw_pointer_cast(movement_counters.data()),
				g.total_weight, bucketSizePerWarp,
				thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem + nrCforBlkSMem + nrCforWrp + nrC_N_leq32 + nrC_N_leq16 + nrC_N_leq8,
				nrC_N_leq4, thrust::raw_pointer_cast(devPrimes.data()), nb_prime,
				thrust::raw_pointer_cast(cardinalityOfComms.data()),
				thrust::raw_pointer_cast(cardinalityOfComms_new.data()),
				wrpSz, thrust::raw_pointer_cast(wDegs.data()));

		//print_vector(in, "in (*): ");

		report_time(start, stop, "neigh_comm ( <=4)");
		//nb_moves = nb_moves + thrust::reduce(movement_counters.begin(), movement_counters.begin() + nrC_N_leq4, (int) 0);

		/*
		   changeAssignment<<< nr_of_block, NR_THREAD_PER_BLOCK>>>(  
		   thrust::raw_pointer_cast(n2c.data()), // change from
		   thrust::raw_pointer_cast(n2c_new.data()), // change to
		   thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem + nrCforBlkSMem + nrCforWrp + nrC_N_leq32 + nrC_N_leq16 + nrC_N_leq8 ,
		   nrC_N_leq4);
		 */

		if (isGauss) {
			if (isToUpdate) {
				nr_of_block = (community_size + NR_THREAD_PER_BLOCK - 1) / NR_THREAD_PER_BLOCK;
				update << <nr_of_block, NR_THREAD_PER_BLOCK>>>(community_size,
						thrust::raw_pointer_cast(tot.data()),
						thrust::raw_pointer_cast(tot_new.data()),
						thrust::raw_pointer_cast(n2c.data()),
						thrust::raw_pointer_cast(n2c_new.data()),
						thrust::raw_pointer_cast(cardinalityOfComms.data()),
						thrust::raw_pointer_cast(cardinalityOfComms_new.data()));
			} else {
				n2c = n2c_new;
				tot = tot_new;
				cardinalityOfComms = cardinalityOfComms_new;
			}
		}
	}


	if (nrC_N_leq32) {

		if (isGauss) {
			//thrust::fill_n(thrust::device, in.begin(), in.size(), 0.0); // initialize in to all zeros '0'
			//tot_new = tot;
			//cardinalityOfComms_new = cardinalityOfComms;
		}

		wrpSz = PHY_WRP_SZ;
		nr_of_block = (nrC_N_leq32 + (NR_THREAD_PER_BLOCK / wrpSz) - 1) / (NR_THREAD_PER_BLOCK / wrpSz);

		bucketSizePerWarp = 61; //MUST BE PRIME
		sizeHashMem = (NR_THREAD_PER_BLOCK / wrpSz) * bucketSizePerWarp * sizeof (HashItem);

		cudaEventRecord(start, 0);

		neigh_comm << < nr_of_block, NR_THREAD_PER_BLOCK, sizeHashMem >>>(
				community_size,
				thrust::raw_pointer_cast(g.indices.data()),
				thrust::raw_pointer_cast(g.links.data()),
				thrust::raw_pointer_cast(g.weights.data()),
				thrust::raw_pointer_cast(n2c.data()),
				thrust::raw_pointer_cast(in.data()),
				thrust::raw_pointer_cast(tot.data()), g.type,
				thrust::raw_pointer_cast(n2c_new.data()),
				thrust::raw_pointer_cast(tot_new.data()),
				thrust::raw_pointer_cast(movement_counters.data()),
				g.total_weight, bucketSizePerWarp,
				thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem + nrCforBlkSMem + nrCforWrp,
				nrC_N_leq32, thrust::raw_pointer_cast(devPrimes.data()), nb_prime,
				thrust::raw_pointer_cast(cardinalityOfComms.data()),
				thrust::raw_pointer_cast(cardinalityOfComms_new.data()),
				wrpSz, thrust::raw_pointer_cast(wDegs.data()));

		report_time(start, stop, "neigh_comm(<=32)");
		//nb_moves = nb_moves + thrust::reduce(movement_counters.begin(), movement_counters.begin() + nrC_N_leq32, (int) 0);

		if (0) {
			changeAssignment << < nr_of_block, NR_THREAD_PER_BLOCK>>>(
					thrust::raw_pointer_cast(n2c.data()), // change from
					thrust::raw_pointer_cast(n2c_new.data()), // change to
					thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem + nrCforBlkSMem + nrCforWrp,
					nrC_N_leq32);
		}

		if (isGauss) {
			if (isToUpdate) {
				nr_of_block = (community_size + NR_THREAD_PER_BLOCK - 1) / NR_THREAD_PER_BLOCK;
				update << <nr_of_block, NR_THREAD_PER_BLOCK>>>(community_size,
						thrust::raw_pointer_cast(tot.data()),
						thrust::raw_pointer_cast(tot_new.data()),
						thrust::raw_pointer_cast(n2c.data()),
						thrust::raw_pointer_cast(n2c_new.data()),
						thrust::raw_pointer_cast(cardinalityOfComms.data()),
						thrust::raw_pointer_cast(cardinalityOfComms_new.data()));
			} else {
				n2c = n2c_new;
				tot = tot_new;
				cardinalityOfComms = cardinalityOfComms_new;
			}
		}
	}


	if (nrCforWrp) {

		if (isGauss) {
			//thrust::fill_n(thrust::device, in.begin(), in.size(), 0.0); // initialize in to all zeros '0'
			//tot_new = tot;
			//cardinalityOfComms_new = cardinalityOfComms;
		}


		wrpSz = PHY_WRP_SZ;
		nr_of_block = (nrCforWrp + (NR_THREAD_PER_BLOCK / wrpSz) - 1) / (NR_THREAD_PER_BLOCK / wrpSz);

		bucketSizePerWarp = WARP_TABLE_SIZE_1;
		sizeHashMem = (NR_THREAD_PER_BLOCK / wrpSz) * bucketSizePerWarp * sizeof (HashItem);

		cudaEventRecord(start, 0);

		neigh_comm << < nr_of_block, NR_THREAD_PER_BLOCK, sizeHashMem >>>(
				community_size,
				thrust::raw_pointer_cast(g.indices.data()),
				thrust::raw_pointer_cast(g.links.data()),
				thrust::raw_pointer_cast(g.weights.data()),
				thrust::raw_pointer_cast(n2c.data()),
				thrust::raw_pointer_cast(in.data()),
				thrust::raw_pointer_cast(tot.data()), g.type,
				thrust::raw_pointer_cast(n2c_new.data()),
				thrust::raw_pointer_cast(tot_new.data()),
				thrust::raw_pointer_cast(movement_counters.data()),
				g.total_weight, bucketSizePerWarp,
				thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem + nrCforBlkSMem,
				nrCforWrp, thrust::raw_pointer_cast(devPrimes.data()), nb_prime,
				thrust::raw_pointer_cast(cardinalityOfComms.data()),
				thrust::raw_pointer_cast(cardinalityOfComms_new.data()),
				wrpSz, thrust::raw_pointer_cast(wDegs.data()));

		report_time(start, stop, "neigh_comm");
		//nb_moves = thrust::reduce(movement_counters.begin(), movement_counters.begin() + nrCforWrp, (int) 0);

		// change community assignment of processed vertices
		if (0) {
			changeAssignment << < nr_of_block, NR_THREAD_PER_BLOCK>>>(
					thrust::raw_pointer_cast(n2c.data()), // change from
					thrust::raw_pointer_cast(n2c_new.data()), // change to
					thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem + nrCforBlkSMem, // of these communities
					nrCforWrp);
		}

		if (isGauss) {
			if (isToUpdate) {
				nr_of_block = (community_size + NR_THREAD_PER_BLOCK - 1) / NR_THREAD_PER_BLOCK;
				update << <nr_of_block, NR_THREAD_PER_BLOCK>>>(community_size,
						thrust::raw_pointer_cast(tot.data()),
						thrust::raw_pointer_cast(tot_new.data()),
						thrust::raw_pointer_cast(n2c.data()),
						thrust::raw_pointer_cast(n2c_new.data()),
						thrust::raw_pointer_cast(cardinalityOfComms.data()),
						thrust::raw_pointer_cast(cardinalityOfComms_new.data()));
			} else {
				n2c = n2c_new;
				tot = tot_new;
				cardinalityOfComms = cardinalityOfComms_new;
			}
		}
	}

	if (nrCforBlkSMem > 0) {


		if (isGauss) {
			//thrust::fill_n(thrust::device, in.begin(), in.size(), 0.0); // initialize in to all zeros '0'
			//tot_new = tot;
			//cardinalityOfComms_new = cardinalityOfComms;
		}

		wrpSz = PHY_WRP_SZ;
		cudaEventRecord(start, 0);

		//std::cout<<" nrBlockForLargeNhoods :"<<nrBlockForLargeNhoods<<"   nrCforBlkSMem: "<<   nrCforBlkSMem<<std::endl;

		lookAtNeigboringComms << <nrBlockForLargeNhoods, NR_THREAD_PER_BLOCK>>>(
				thrust::raw_pointer_cast(g.indices.data()),
				thrust::raw_pointer_cast(g.links.data()),
				thrust::raw_pointer_cast(g.weights.data()),
				thrust::raw_pointer_cast(n2c.data()),
				thrust::raw_pointer_cast(in.data()),
				thrust::raw_pointer_cast(tot.data()), g.type,
				thrust::raw_pointer_cast(n2c_new.data()),
				NULL,
				thrust::raw_pointer_cast(tot_new.data()),
				thrust::raw_pointer_cast(moveCounters.data()), g.total_weight,
				thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem, nrCforBlkSMem,
				thrust::raw_pointer_cast(globalHashTable.data()),
				thrust::raw_pointer_cast(hashTablePtrs.data()),
				thrust::raw_pointer_cast(devPrimes.data()), nb_prime, wrpSz,
				thrust::raw_pointer_cast(cardinalityOfComms.data()),
				thrust::raw_pointer_cast(cardinalityOfComms_new.data()),
				thrust::raw_pointer_cast(wDegs.data()));
		report_time(start, stop, "lookAtNeigboringComms(sh)");
		//nb_moves = nb_moves + thrust::reduce(moveCounters.begin(), moveCounters.begin() + nrBlockForLargeNhoods, (int) 0);
		/*
		   if (0) {
		   changeAssignment << < nr_of_block, NR_THREAD_PER_BLOCK>>>(
		   thrust::raw_pointer_cast(n2c.data()), // change from
		   thrust::raw_pointer_cast(n2c_new.data()), // change to
		   thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGMem, nrCforBlkSMem);
		   }
		 */

		if (isGauss) {
			if (isToUpdate) {
				nr_of_block = (community_size + NR_THREAD_PER_BLOCK - 1) / NR_THREAD_PER_BLOCK;
				update << <nr_of_block, NR_THREAD_PER_BLOCK>>>(community_size,
						thrust::raw_pointer_cast(tot.data()),
						thrust::raw_pointer_cast(tot_new.data()),
						thrust::raw_pointer_cast(n2c.data()),
						thrust::raw_pointer_cast(n2c_new.data()),
						thrust::raw_pointer_cast(cardinalityOfComms.data()),
						thrust::raw_pointer_cast(cardinalityOfComms_new.data()));
			} else {
				n2c = n2c_new;
				tot = tot_new;
				cardinalityOfComms = cardinalityOfComms_new;
			}
		}
	}

#ifdef LARGE_LATER

	if (nrCforBlkGMem > 0) {

		if (isGauss) {
			//thrust::fill_n(thrust::device, in.begin(), in.size(), 0.0); // initialize in to all zeros '0'
			//tot_new = tot;
			//cardinalityOfComms_new = cardinalityOfComms;
		}

		wrpSz = PHY_WRP_SZ;

		cudaEventRecord(start, 0);

		//std::cout<<" nrBlockForLargeNhoods: "<<nrBlockForLargeNhoods<<" nrCforBlkGMem:  "<<  nrCforBlkGMem<<std::endl;
		lookAtNeigboringComms << <nrBlockForLargeNhoods, (NR_THREAD_PER_BLOCK * 2)>>>(
				thrust::raw_pointer_cast(g.indices.data()),
				thrust::raw_pointer_cast(g.links.data()),
				thrust::raw_pointer_cast(g.weights.data()),
				thrust::raw_pointer_cast(n2c.data()),
				thrust::raw_pointer_cast(in.data()),
				thrust::raw_pointer_cast(tot.data()), g.type,
				thrust::raw_pointer_cast(n2c_new.data()),
				NULL,
				thrust::raw_pointer_cast(tot_new.data()),
				thrust::raw_pointer_cast(moveCounters.data()), g.total_weight,
				thrust::raw_pointer_cast(g_next.indices.data()), nrCforBlkGMem,
				thrust::raw_pointer_cast(globalHashTable.data()),
				thrust::raw_pointer_cast(hashTablePtrs.data()),
				thrust::raw_pointer_cast(devPrimes.data()), nb_prime, wrpSz,
				thrust::raw_pointer_cast(cardinalityOfComms.data()),
				thrust::raw_pointer_cast(cardinalityOfComms_new.data()),
				thrust::raw_pointer_cast(wDegs.data()));

		report_time(start, stop, "lookAtNeigboringComms");
		//nb_moves = nb_moves + thrust::reduce(moveCounters.begin(), moveCounters.begin() + nrBlockForLargeNhoods, (int) 0);

		/*
		   if (0) {
		   changeAssignment << < nr_of_block, NR_THREAD_PER_BLOCK>>>(
		   thrust::raw_pointer_cast(n2c.data()), // change from
		   thrust::raw_pointer_cast(n2c_new.data()), // change to
		   thrust::raw_pointer_cast(g_next.indices.data()), nrCforBlkGMem);
		   }
		 */

		if (isGauss) {
			if (isToUpdate) {
				nr_of_block = (community_size + NR_THREAD_PER_BLOCK - 1) / NR_THREAD_PER_BLOCK;
				update << <nr_of_block, NR_THREAD_PER_BLOCK>>>(community_size,
						thrust::raw_pointer_cast(tot.data()),
						thrust::raw_pointer_cast(tot_new.data()),
						thrust::raw_pointer_cast(n2c.data()),
						thrust::raw_pointer_cast(n2c_new.data()),
						thrust::raw_pointer_cast(cardinalityOfComms.data()),
						thrust::raw_pointer_cast(cardinalityOfComms_new.data()));
			} else {
				n2c = n2c_new;
				tot = tot_new;
				cardinalityOfComms = cardinalityOfComms_new;
			}
		}
	}
#endif

	/*
	   if (isGauss) {

	//thrust::fill_n(thrust::device, in.begin(), in.size(), 0.0); // initialize in to all zeros '0'

	wrpSz = PHY_WRP_SZ;
	load_per_blk = CHUNK_PER_WARP * (NR_THREAD_PER_BLOCK / wrpSz);
	nr_of_block = (community_size + load_per_blk - 1) / load_per_blk;

	//void computeInternals(int *indices, unsigned int *links, float *weights, int *n2c, float *in, unsigned int nrComms);

	computeInternals << <nr_of_block, NR_THREAD_PER_BLOCK>>>(thrust::raw_pointer_cast(g.indices.data()),
	thrust::raw_pointer_cast(g.links.data()),
	thrust::raw_pointer_cast(g.weights.data()),
	thrust::raw_pointer_cast(n2c.data()),
	thrust::raw_pointer_cast(in.data()), community_size, g.type);

	}*/

	//std::cout << "---------Now  " << nb_moves << std::endl;

	/*
	   std::cout << "#Moves (Total):" << nb_moves << " nrCforBlkSMem : " << nrCforBlkSMem << " nrCforBlkGMem :" << nrCforBlkGMem << std::endl;

	   if (0) {
	   std::cout << "After Traversing.............. " << std::endl;
	   print_vector(in, "IN:");
	   print_vector(tot, "TOT:");

	   print_vector(n2c_new, " n2c_new : ");
	   print_vector(tot_new, " tot_new : ");
	   }
	 */

	/*
	   if (0) {

	   float sum_in = thrust::reduce(in.begin(), in.end(), 0.0);
	//float sum_tot = thrust::reduce(tot.begin(), tot.end(), 0.0);

	thrust::host_vector<float> hvec = tot;

	thrust::host_vector<float> hIN = in;
	double stot = 0;

	for (int i = 0; i < hvec.size(); i++)
	stot += hvec[i] * hvec[i];

	std::cout << "sin:" << sum_in << "	 stot: " << stot << std::endl;
	//std::cout << " IN[0]: "<< hIN[0]<< " IN[1]: "<< hIN[1] << std::endl;
	//std::cout << "sum_in = " << sum_in << " sum_tot = " << sum_tot << std::endl;
	}

	 */
	new_mod = modularity(tot, in);


	double scur_mod = cur_mod;
	double snew_mod = new_mod;
	/*
	   std::cout << nrIteration << " " << "Modularity   " << cur_mod << " --> " << new_mod <<
	   " Gain: " << (new_mod - cur_mod) << std::endl;
	 */
	if ((new_mod - cur_mod) >= threshold) { // Mind this If condition

		n2c_old = n2c;
		n2c = n2c_new;
		tot = tot_new;
		cardinalityOfComms = cardinalityOfComms_new;

		cur_mod = new_mod;

		if (cur_mod < init_mod) {
			cur_mod = init_mod;
		}

		improvement = true;

	} else {
		//std::cout << "Break the loop " << std::endl;
		break;
	}
	if (nrIteration)
		std::cout << nrIteration << " " << "Modularity   " << scur_mod << " --> "
			<< snew_mod << " Gain: " << (snew_mod - scur_mod) << std::endl;


	/*
	   if (verbose) {
	   print_vector(n2c, " n2c (After Swap): ");
	   print_vector(in, " in (After Swap): ");
	   print_vector(tot, " tot (After Swap): ");
	   }
	 */


	tot_new.clear();

	t2 = clock();
	float diff = (float)t2 - (float) t1;
	float seconds = diff / CLOCKS_PER_SEC;
	std::cout<< "iteration "<<(nrIteration+1)<<": "<<seconds<<" sec"<<std::endl;

	} while (++nrIteration < 1000);

	cardinalityOfComms.clear();
	cardinalityOfComms_new.clear();
	globalHashTable.clear();
	hashTablePtrs.clear();

	n2c = n2c_old;
	n2c_old.clear();

	//std::cout<<"#iteration: "<<nrIteration<<std::endl;
	// print_vector(n2c, " n2c (Before contraction)");
	/*
	   thrust::host_vector<int> hn2c = n2c;

	   std::ofstream ofs ("n2c.txt", std::ofstream::out);
	   for(int i=0; i< hn2c.size(); i++) {
	   ofs<<hn2c[i]<<" ";
	   }
	   ofs<<"\n";
	   ofs.close();

	   std::ofstream outfile ("n2c.txt",std::ofstream::binary);
	   outfile.write ((char*)&hn2c[0],sizeof(int)*hn2c.size());
	   char newline= '\n';
	   outfile.write ((char*)&newline,sizeof(char));
	   outfile.close();
	 */

	tot_new.clear();
	g_next.indices.clear();
	g_next.links.clear();
	n2c_new.clear(); // <-----------
	wDegs.clear();
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	return cur_mod;
}

