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
#include"hostconstants.h"
#include"thrust/reduce.h"
#include"thrust/count.h"
#include"fstream"
#include <thrust/gather.h>

void Community::compute_next_graph(cudaStream_t *streams, int nrStreams,
		cudaEvent_t &start, cudaEvent_t &stop) {

	//std::cout << "\nCompute_next_graph() \n";

	int new_nb_comm = g_next.nb_nodes;

	bool hostPrint = false;
	int sc;
	sc = 0; //std::cin>>sc;
	hostPrint = (sc > 1);


	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);



	//Save a copy of "pos_ptr_of_new_comm"

	thrust::device_vector<int> super_node_ptrs(pos_ptr_of_new_comm);
	/*
	   if (hostPrint) {
	   print_vector(super_node_ptrs, "Super Node Ptrs: ");
	   print_vector(n2c_new, " n2c_new; before group_nodes_based_on_new_CID");
	   }
	 */
	//Place nodes of same community together

	comm_nodes.resize(g.nb_nodes);
	cudaEventRecord(start, 0);
	int load_per_blk = CHUNK_PER_WARP * (NR_THREAD_PER_BLOCK / PHY_WRP_SZ);
	int nr_of_block = (community_size + load_per_blk - 1) / load_per_blk;

	group_nodes_based_on_new_CID << < nr_of_block, NR_THREAD_PER_BLOCK>>>
		(thrust::raw_pointer_cast(comm_nodes.data()),
		 thrust::raw_pointer_cast(pos_ptr_of_new_comm.data()),
		 thrust::raw_pointer_cast(n2c_new.data()),
		 thrust::raw_pointer_cast(n2c.data()), g.nb_nodes);


	report_time(start, stop, "group_nodes_based_on_new_CID");

	this->pos_ptr_of_new_comm.clear();

	//////////////////////////////////////////////////////////////////////////////

	thrust::device_vector<int> degree_per_node(super_node_ptrs.size());
	thrust::transform(thrust::device, super_node_ptrs.begin() + 1, super_node_ptrs.end(), super_node_ptrs.begin(),
			degree_per_node.begin(), thrust::minus<int>());



	int largestCommSize = *thrust::max_element(degree_per_node.begin(), degree_per_node.end());


	//std::cout << "-----------------------------------largestCommSize: " << largestCommSize << std::endl;


	/////////////////////////////////////////////////////////////////////

	//print_vector(comm_nodes, " comm_nodes : ");

	/*
	   if(0){

	   thrust::host_vector<int> allnodes = comm_nodes;

	//unsigned int lastone = 777;
	for (unsigned int i = 0; i < g.nb_nodes; i++) {

	if(lastone != n2c[allnodes[i]]){

	std::cout<<std::endl; 
	}
	lastone = n2c[allnodes[i]];

	if( n2c_new[n2c[allnodes[i]]] ==63 )  std::cout << allnodes[i] << ":" << n2c[ allnodes[i]] <<":" << n2c_new[n2c[allnodes[i]]] <<"  ";
	}	
	}
	 */
	/*
	   if (0) {

	   std::cout << std::endl << "Node:Community " << std::endl;
	   for (int i = 0; i < g.nb_nodes; i++) {
	   if( (i >=320 && i <=329) || (i >= 2945 && i <= 2949) ) std::cout << i << "(" << n2c_new[n2c[i]] << ") ";
	   }
	   std::cout << std::endl;
	   }

	   if (hostPrint) {
	   print_vector(super_node_ptrs, "Super Node Ptrs: ");
	   }
	 */



	// construct next Graph
	g_next.indices.resize(new_nb_comm + 1);


	//-------Estimate the size of neighborhood of each new community------//

	thrust::device_vector<int> estimatedSizeOfNeighborhoods(new_nb_comm + 1, -1);

	unsigned int wrpSz = PHY_WRP_SZ; //1;
	int nr_block_needed = (new_nb_comm + (NR_THREAD_PER_BLOCK / wrpSz) - 1) / (NR_THREAD_PER_BLOCK / wrpSz);

	cudaEventRecord(start, 0);
	computeBoundOfNeighoodSize << < nr_block_needed, NR_THREAD_PER_BLOCK>>>(
			thrust::raw_pointer_cast(super_node_ptrs.data()),
			thrust::raw_pointer_cast(g.indices.data()),
			thrust::raw_pointer_cast(comm_nodes.data()), new_nb_comm,
			thrust::raw_pointer_cast(estimatedSizeOfNeighborhoods.data()),
			wrpSz);
	report_time(start, stop, "estimate_size_of_neighborhoods");

	/*
	   if (hostPrint) {
	   print_vector(estimatedSizeOfNeighborhoods, "estimatedSizeOfNeighborhoods: ");
	   }
	 */

	//
	/*
	   if (0) {
	   thrust::host_vector<int> commNodesHost = comm_nodes;
	   thrust::host_vector<int> superPtrsHost = super_node_ptrs;

	   for (int i = 0; i < superPtrsHost.size() - 1; i++) {
	   if(i==63){
	   thrust::sort(commNodesHost.begin() + superPtrsHost[i], commNodesHost.begin() + superPtrsHost[i + 1]);

	   std::cout << "RNR ";
	   for (int j = superPtrsHost[i]; j < superPtrsHost[i + 1]; j++) {
	   std::cout << commNodesHost[j] << " ";
	   }
	   std::cout << std::endl;
	   }
	   }
	   commNodesHost.clear();
	   superPtrsHost.clear();
	   }
	 */
	cudaEventRecord(start, 0);
	unsigned int bucketSizePerWarp = WARP_TABLE_SIZE_1;

	IsGreaterThanLimit<int, int> filterForBlkGMem(SHARED_TABLE_SIZE);
	IsInRange<int, int> filterForBlkSMem(WARP_TABLE_SIZE_1 + 1, SHARED_TABLE_SIZE);
	IsInRange<int, int> filterForWrp(0, WARP_TABLE_SIZE_1);


	//------------Filter communities to be processed by block based on upper bound---------//

	//Count First

	int nrCforBlkGbMem = thrust::count_if(thrust::device,
			estimatedSizeOfNeighborhoods.begin(),
			estimatedSizeOfNeighborhoods.end(), filterForBlkGMem);

	int nrCforBlkShMem = thrust::count_if(thrust::device,
			estimatedSizeOfNeighborhoods.begin(),
			estimatedSizeOfNeighborhoods.end(), filterForBlkSMem);

	int nrCforWrp = thrust::count_if(thrust::device,
			estimatedSizeOfNeighborhoods.begin(),
			estimatedSizeOfNeighborhoods.end(), filterForWrp);




	//    std::cout << "#Community Processed By Warp = " << nrCforWrp << std::endl;

	//Lets copy all community ids in g_next.links
	g_next.links.resize(new_nb_comm, 0);
	thrust::sequence(g_next.links.begin(), g_next.links.end(), 0);

	//Use g_next.indices to copy community ids with decreasing sizes of neighborhood
	g_next.indices.resize(new_nb_comm, -1);

	//Community ids with larger UpperBound on SoN first

	thrust::copy_if(thrust::device, g_next.links.begin(),
			g_next.links.end(), estimatedSizeOfNeighborhoods.begin(),
			g_next.indices.begin(), filterForBlkGMem);

	//^^copied first "nrCforBlkGbMem"

	thrust::copy_if(thrust::device, g_next.links.begin(),
			g_next.links.end(), estimatedSizeOfNeighborhoods.begin(),
			g_next.indices.begin() + nrCforBlkGbMem, filterForBlkSMem);

	//^^copied next "nrCforBlkShMem"

	//Then community ids with smaller UpperBound on SoN
	thrust::copy_if(thrust::device, g_next.links.begin(), g_next.links.end(),
			estimatedSizeOfNeighborhoods.begin(),
			g_next.indices.begin() + nrCforBlkGbMem + nrCforBlkShMem,
			filterForWrp);

	/*
	   std::cout << "new_nb_comm = " << new_nb_comm << std::endl;
	   std::cout << "nrCforBlkGbMem = " << nrCforBlkGbMem << std::endl;
	   std::cout << "nrCforBlkShMem = " << nrCforBlkShMem << std::endl;
	   std::cout << "nrCforWrp = " << nrCforWrp << std::endl;
	 */
	assert((nrCforBlkGbMem + nrCforBlkShMem + nrCforWrp) == new_nb_comm);
	/*
	   if (0) {
	   thrust::host_vector<int> esSizes = estimatedSizeOfNeighborhoods;
	   thrust::host_vector<int> bigCommunites = g_next.indices;
	   for (int k = 0; k < thrust::min<int>(5, bigCommunites.size()); k++) {
	   std::cout << bigCommunites[k] << ":::" << esSizes[bigCommunites[k]] << std::endl;
	   }
	   esSizes.clear();
	   bigCommunites.clear();
	   }*/

	// Now, use g_next.links to copy sizes of neighborhood according to order given by g_next.indices

	g_next.links.resize(g_next.indices.size(), 0);

	thrust::gather(thrust::device, g_next.indices.begin(),
			g_next.indices.end(), estimatedSizeOfNeighborhoods.begin(),
			g_next.links.begin());
	report_time(start, stop, "FilterGather");

	//    std::cout<<"Gathered\n";
	//  std::cin>>sc;
	/*
	   if (0) {
	   thrust::host_vector<unsigned int> gnlinks = g_next.links;
	   thrust::host_vector<int> bigCommunites = g_next.indices;

	   for (int i = 0; i < thrust::min(5, nrCforBlkGbMem); i++) {
	   std::cout << bigCommunites[i] << "*" << gnlinks[i] << std::endl;
	   }

	   gnlinks.clear();
	   bigCommunites.clear();
	   }
	 */


	sc = 0; //std::cin>>sc;
	//Sort according to size of neighborhood ; only first nrCforBlkGbMem
	if ((nrCforBlkGbMem + nrCforBlkShMem) > 0) {

		int sortLen = nrCforBlkGbMem + nrCforBlkShMem;

		//std::cout<<"Sorting "<<sortLen <<" entries"<<std::endl;

		thrust::sort_by_key(g_next.links.begin(), g_next.links.begin() + sortLen, g_next.indices.begin(), thrust::greater<unsigned int>());
	}


	/*
	   if (1) {
	   thrust::host_vector<int> esSizes = estimatedSizeOfNeighborhoods;
	   thrust::host_vector<int> bigCommunites = g_next.indices;

	   for (int k = 0; k < thrust::min<int>(bigCommunites.size(), 8); k++) {
	   std::cout << bigCommunites[k] << "::" << esSizes[bigCommunites[k]] << std::endl;
	   }
	   esSizes.clear();
	   bigCommunites.clear();
	   }
	 */

	sc = 0; //std::cin>>sc;
	int nrBlockForLargeNhoods = 150;

	nrBlockForLargeNhoods = thrust::min(thrust::max(nrCforBlkGbMem, nrCforBlkShMem), nrBlockForLargeNhoods);

	thrust::device_vector<int> hashTablePtrs(nrBlockForLargeNhoods + 1, 0);
	//////////////////////////////////////////////////
	//void preComputePrimes(int *primes,int nrPrimes, int* thresholds, int nrBigBlock, int *selectedPrimes,  int WARP_SIZE);

	wrpSz = PHY_WRP_SZ;
	;

	nr_block_needed = (nrBlockForLargeNhoods + (NR_THREAD_PER_BLOCK / wrpSz) - 1) / (NR_THREAD_PER_BLOCK / wrpSz);
	if (nrBlockForLargeNhoods > 0)
		preComputePrimes << < nr_block_needed, NR_THREAD_PER_BLOCK >>> (thrust::raw_pointer_cast(devPrimes.data()),
				nb_prime, thrust::raw_pointer_cast(g_next.links.data()), nrBlockForLargeNhoods,
				thrust::raw_pointer_cast(hashTablePtrs.data()) + 1, wrpSz);

	/*
	   if (1) {
	   thrust::host_vector<int> esSizes = hashTablePtrs;   
	   thrust::host_vector<unsigned int> thresholds=  g_next.links;
	   int spaceReq=0;
	   for (int k = 0; k < nrBlockForLargeNhoods; k++) {
	   std::cout << thresholds[k]<<" nearest Prime -> " <<esSizes[k+1]<<std::endl;
	   if(thresholds[k]> esSizes[k+1])
	   std::cout<<"PROBLEM in HOST, call to prime computation"<<std::endl;
	   spaceReq += esSizes[k+1];
	   }

	   std::cout<< "Total Space requried: "<< spaceReq <<std::endl;

	   esSizes.clear();
	   }
	 */

	thrust::inclusive_scan(hashTablePtrs.begin(),
			hashTablePtrs.begin() + (nrBlockForLargeNhoods + 1),
			hashTablePtrs.begin(), thrust::plus<int>());


	///////////////////////////////////////////////////

	//g_next.links contains sizes of big neighborhoods

	/*--------------->
	  thrust::inclusive_scan(g_next.links.begin(),
	  g_next.links.begin() + nrBlockForLargeNhoods,
	  hashTablePtrs.begin() + 1, thrust::plus<int>());
	 */
	/********************/
	/*
	   if (1) {
	   thrust::host_vector<int> esSizes = hashTablePtrs;   

	   for (int k = 0; k < nrBlockForLargeNhoods; k++) {
	   std::cout << esSizes[k]<<" ";
	   }
	   std::cout<<std::endl;
	   esSizes.clear();
	   }
	 */

	//------->thrust::transform( hashTablePtrs.begin(), hashTablePtrs.end(), hashTablePtrs.begin(), hashTablePtrs.begin(),thrust::plus<int>());
	/*   
	     if (1) {
	     thrust::host_vector<int> esSizes = hashTablePtrs;   

	     for (int k = 0; k < nrBlockForLargeNhoods/8; k++) {
	     std::cout <<k<<":"<< esSizes[k]<<" ";
	     }
	     std::cout<<std::endl;
	     esSizes.clear();
	     }
	 */

	//thrust::transform( hashTablePtrs.begin(), hashTablePtrs.end(), hashTablePtrs.begin(), hashTablePtrs.begin(),thrust::plus<int>());

	/*
	   if (0) {
	   thrust::host_vector<int> esSizes = hashTablePtrs;   

	   for (int k = 0; k < nrBlockForLargeNhoods; k++) {
	   std::cout << esSizes[k]<<" ";
	   }
	   std::cout<<std::endl;
	   esSizes.clear();
	   }
	 */
	thrust::device_vector<HashItem> globalHashTable(hashTablePtrs.back());
	/*********************/
	// thrust::device_vector<HashItem> globalHashTable(3 * hashTablePtrs.back());

	int szHTmem = thrust::reduce(g_next.links.begin(),
			g_next.links.begin() + nrBlockForLargeNhoods, (int) 0);

	//std::cout << globalHashTable.size() << ":" << 2 * szHTmem << std::endl;

	//-------Prefix sum on estimate the size of neighborhoods to determine global positions for new communities-----//

	cudaEventRecord(start, 0);

	thrust::exclusive_scan(thrust::device, estimatedSizeOfNeighborhoods.begin(),
			estimatedSizeOfNeighborhoods.end(), estimatedSizeOfNeighborhoods.begin(),
			(int) 0, thrust::plus<int>());

	report_time(start, stop, "thrust::exclusive_scan");

	int upperBoundonTotalSize = estimatedSizeOfNeighborhoods.back();

	/*
	   if (hostPrint) {
	   print_vector(estimatedSizeOfNeighborhoods, "estimatedSizeOfNeighborhoods: ");
	   }
	   std::cout << "Before big allocation; UpperBoundOnTotalSize = " << upperBoundonTotalSize << std::endl;
	 */


	//--------------Allocate memory for new links and weights-------------//


	thrust::device_vector<unsigned int> member_count_per_new_comm(new_nb_comm + 1, 0); // exact count

	thrust::device_vector<unsigned int> new_nighbor_lists(upperBoundonTotalSize);
	thrust::device_vector<float> new_weight_lists(upperBoundonTotalSize);


	//std::cout << "nrBlockForLargeNhoods: " << nrBlockForLargeNhoods << std::endl;
	/*
	   if (hostPrint) {
	   print_vector(member_count_per_new_comm, "Exact #neighbor per community:");
	   }

	 */
	wrpSz = PHY_WRP_SZ;
	cudaEventRecord(start, 0);
	if (nrCforBlkGbMem > 0)
		findNewNeighodByBlock << < nrBlockForLargeNhoods, NR_THREAD_PER_BLOCK/*, 0, streams[0]*/>>>
			(thrust::raw_pointer_cast(super_node_ptrs.data()),
			 thrust::raw_pointer_cast(new_weight_lists.data()),
			 thrust::raw_pointer_cast(new_nighbor_lists.data()),
			 thrust::raw_pointer_cast(member_count_per_new_comm.data()), // puts zero in position zero for prefix sum
			 thrust::raw_pointer_cast(g.indices.data()),
			 thrust::raw_pointer_cast(g.weights.data()),
			 thrust::raw_pointer_cast(g.links.data()),
			 thrust::raw_pointer_cast(comm_nodes.data()), new_nb_comm,
			 thrust::raw_pointer_cast(n2c.data()),
			 thrust::raw_pointer_cast(n2c_new.data()),
			 thrust::raw_pointer_cast(estimatedSizeOfNeighborhoods.data()),
			 g.type, bucketSizePerWarp,
			 thrust::raw_pointer_cast(g_next.indices.data()), //int* candidateComms
			 nrCforBlkGbMem, //int nrCandidateComms
			 thrust::raw_pointer_cast(globalHashTable.data()),
			 thrust::raw_pointer_cast(hashTablePtrs.data()),
			 thrust::raw_pointer_cast(devPrimes.data()), nb_prime, wrpSz);

	report_time(start, stop, "findNewNeighodByBlock(GlobalMemory)");

	/*
	   int sum_MC = thrust::reduce(member_count_per_new_comm.begin(), member_count_per_new_comm.end(), (int) 0);
	   std::cout << "sum_MC(blkGlb): " << sum_MC << std::endl;
	 */

	sc = 0; //std::cin>>sc;
	cudaEventRecord(start, 0);
	if (nrCforBlkShMem > 0)
		findNewNeighodByBlock << < nrBlockForLargeNhoods, NR_THREAD_PER_BLOCK/*, 0, streams[1]*/>>>
			(thrust::raw_pointer_cast(super_node_ptrs.data()),
			 thrust::raw_pointer_cast(new_weight_lists.data()),
			 thrust::raw_pointer_cast(new_nighbor_lists.data()),
			 thrust::raw_pointer_cast(member_count_per_new_comm.data()), // puts zero in position zero for prefix sum
			 thrust::raw_pointer_cast(g.indices.data()),
			 thrust::raw_pointer_cast(g.weights.data()),
			 thrust::raw_pointer_cast(g.links.data()),
			 thrust::raw_pointer_cast(comm_nodes.data()), new_nb_comm,
			 thrust::raw_pointer_cast(n2c.data()),
			 thrust::raw_pointer_cast(n2c_new.data()),
			 thrust::raw_pointer_cast(estimatedSizeOfNeighborhoods.data()),
			 g.type, bucketSizePerWarp,
			 thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGbMem, //int* candidateComms
			 nrCforBlkShMem, //int nrCandidateComms
			 thrust::raw_pointer_cast(globalHashTable.data()),
			 thrust::raw_pointer_cast(hashTablePtrs.data()),
			 thrust::raw_pointer_cast(devPrimes.data()), nb_prime, wrpSz);

	report_time(start, stop, "findNewNeighodByBlock(SharedMemory)");
	/*
	   sum_MC = thrust::reduce(member_count_per_new_comm.begin(), member_count_per_new_comm.end(), (int) 0);

	   std::cout << "sum_MC(blkShrd): " << sum_MC << std::endl;
	 */





	//------------Compute neighborhood of new communities-----------------//

	sc = 0; //std::cin>>sc;

	//if( new_nb_comm!=12525 ){

	sc = 0; //std::cin>>sc;

	//std::cout << "Pre: nr_block_needed:" << nr_block_needed << std::endl;
	wrpSz = PHY_WRP_SZ;
	nr_block_needed = (nrCforWrp + (NR_THREAD_PER_BLOCK / wrpSz) - 1) / (NR_THREAD_PER_BLOCK / wrpSz);
	//std::cout << "Post: nr_block_needed:" << nr_block_needed << std::endl;

	nr_block_needed = thrust::min(nr_block_needed, 1920);

	unsigned int sharedMemSzPerBlock = WARP_TABLE_SIZE_1 * sizeof (HashItem) * NR_THREAD_PER_BLOCK / wrpSz;

	cudaEventRecord(start, 0);
	if (nrCforWrp)
		determineNewNeighborhood << < nr_block_needed, NR_THREAD_PER_BLOCK, sharedMemSzPerBlock/*, streams[2]*/>>>
			(thrust::raw_pointer_cast(super_node_ptrs.data()),
			 thrust::raw_pointer_cast(new_weight_lists.data()),
			 thrust::raw_pointer_cast(new_nighbor_lists.data()),
			 thrust::raw_pointer_cast(member_count_per_new_comm.data()), // puts zero in position zero for prefix sum
			 thrust::raw_pointer_cast(g.indices.data()),
			 thrust::raw_pointer_cast(g.weights.data()),
			 thrust::raw_pointer_cast(g.links.data()),
			 thrust::raw_pointer_cast(comm_nodes.data()), new_nb_comm,
			 thrust::raw_pointer_cast(n2c.data()),
			 thrust::raw_pointer_cast(n2c_new.data()),
			 thrust::raw_pointer_cast(estimatedSizeOfNeighborhoods.data()),
			 g.type, WARP_TABLE_SIZE_1,
			 thrust::raw_pointer_cast(g_next.indices.data()) + nrCforBlkGbMem + nrCforBlkShMem,
			 nrCforWrp, wrpSz);

	report_time(start, stop, "determine_neighbors_of_new_comms");

	//print_vector(member_count_per_new_comm, "Neighbor Counts( per new community): ", "MC");

	hashTablePtrs.clear();
	globalHashTable.clear();

	estimatedSizeOfNeighborhoods.clear();
	n2c.clear();
	n2c_new.clear();
	comm_nodes.clear();
	/**************** OKAY ?****************/
	g.indices.clear();
	g.links.clear();
	g.weights.clear();
	/********************************/

	std::cout << "#New Community: " << new_nb_comm << std::endl;
	/*
	   sum_MC = thrust::reduce(member_count_per_new_comm.begin(), member_count_per_new_comm.end(), (int) 0);

	   std::cout << "sum_MC(warp): " << sum_MC << std::endl;

	   if (hostPrint) {
	   std::cout << "#New Community: " << new_nb_comm << std::endl;
	   print_vector(new_weight_lists, "New Weights: ");
	   print_vector(new_nighbor_lists, "New Neighbors: ");
	   print_vector(super_node_ptrs, "Super Node Ptrs: ");
	   }
	 */
	//---------Put data accordingly to new graph-------------------//
	g_next.type = WEIGHTED;
	g_next.indices.resize(member_count_per_new_comm.size(), 0);

	thrust::inclusive_scan(thrust::device, member_count_per_new_comm.begin(),
			member_count_per_new_comm.end(), g_next.indices.begin(), thrust::plus<int>());


	member_count_per_new_comm.clear();


	int nr_edges_in_new_graph = g_next.indices.back();
	g_next.nb_links = (unsigned int) nr_edges_in_new_graph;

	//std::cout << "#E(New Graph): " << nr_edges_in_new_graph << std::endl;

	/*if (0) {
	  std::cout << std::endl << "g_next.nb_links: " << g_next.nb_links << std::endl;
	  std::cout << g_next.links.size() << ":" << g_next.weights.size() << std::endl;
	  }
	 */


	//Filter out unused spaces from global memory and copy to g_next
	g_next.links.resize(g_next.nb_links);
	g_next.weights.resize(g_next.nb_links);

	sc = 0; //std::cin>>sc;

	int nrElmntToCpy = thrust::count_if(thrust::device, new_nighbor_lists.begin(), new_nighbor_lists.end(),
			IsLessLimit<unsigned int, unsigned int>((unsigned int) new_nb_comm));


	//std::cout << "NNS: " << new_nighbor_lists.size() << " , #E2C = " << nrElmntToCpy << std::endl;

	/*
	   if (hostPrint) {
	   std::cout << "-----------------------WE NE--------------" << std::endl;
	   print_vector(new_weight_lists, "WE:");
	   print_vector(new_nighbor_lists, "NE:");

	   }

	 */
	thrust::copy_if(thrust::device, new_nighbor_lists.begin(),
			new_nighbor_lists.end(), g_next.links.begin(),
			IsLessLimit<unsigned int, unsigned int>((unsigned int) new_nb_comm));

	new_nighbor_lists.clear();

	thrust::copy_if(thrust::device, new_weight_lists.begin(),
			new_weight_lists.end(), g_next.weights.begin(),
			Is_Non_Negative<float, float>());

	new_weight_lists.clear();
	/*
	   std::cin>>sc;


	   if (hostPrint) {
	   std::cout << std::endl << "#Node: " << g_next.nb_nodes << std::endl;
	   std::cout << std::endl << "#Edge: " << g_next.nb_links << std::endl;

	   print_vector(g_next.indices, "Copied Indices: ");
	   print_vector(g_next.links, "Copied NE: ");
	   print_vector(g_next.weights, "Copied WE: ");
	   }
	 */
	/*if (0) {
	  thrust::host_vector<unsigned int> gnlinks = g_next.links;
	  thrust::host_vector<int> gnIndices = g_next.indices;
	  thrust::host_vector<float> gnWeights = g_next.weights;

	  std::ofstream ofs;
	  ofs.open ("n2c.txt", std::ofstream::out | std::ofstream::app);

	  for (unsigned int i = 0; i < new_nb_comm; i++) {

	  unsigned int startNbr = gnIndices[i];
	  unsigned int endNbr = gnIndices[i + 1];
	//thrust::sort(gnlinks.begin() + startNbr, gnlinks.begin() + endNbr);

	thrust::sort_by_key(gnlinks.begin() + startNbr, gnlinks.begin() + endNbr, gnWeights.begin() + startNbr);

	//std::cout << (i+1) <<"[" << (endNbr -startNbr) << "]"<< ":";
	ofs<<i<<": ";

	for (unsigned int j = startNbr; j < endNbr; j++) {
	//std::cout << " " << (gnlinks[j]+1)<<"("<<gnWeights[j]<<")";
	ofs<< (gnlinks[j])<<"("<<gnWeights[j]<<")"<<" ";
	}
	ofs<<"\n";
	//std::cout << std::endl;

	}
	ofs.close();
	}*/

	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);

}
