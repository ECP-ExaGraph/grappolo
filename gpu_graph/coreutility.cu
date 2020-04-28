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

#define LOAD_FACTOR 2

//#define CID (1<<29)
//#define WID (1<<29)
//#define EID (1<<29)
//#define DUMP 0
#ifdef RUNONGPU

__device__
#endif
void initByBlock(HashItem* shashTable, unsigned int bucketSize,
        unsigned int workerId, unsigned int stride) {

    __syncthreads();
    for (unsigned int i = workerId; i < bucketSize; i = i + stride) {
        shashTable[i].cId = FLAG_FREE;
        shashTable[i].gravity = 0.0;
    }
    __syncthreads();
}



#ifdef RUNONGPU

__device__
#endif
float blockReduceFloat(float myCounter, unsigned int WARP_SIZE) {



    int laneId = threadIdx.x & (WARP_SIZE - 1);
    int nrWarpInBlock = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    float toReturn = -1.0;

    if (nrWarpInBlock <= PHY_WRP_SZ) {
        //warp reduce
        for (int i = WARP_SIZE / 2; i >= 1; i = i / 2) {
            myCounter += __shfl_xor(myCounter, i, WARP_SIZE);
        }


        // Shared memory to gather result from all warps
        __shared__ float foundByWarps[ PHY_WRP_SZ ];
        volatile float* vmem = foundByWarps;

        /*
        //Initialize to zeros; only by one warp
        if ((threadIdx.x / WARP_SIZE) == 0) {

            for (int i = laneId; i < WARP_SIZE; i = i + WARP_SIZE) {
                vmem[i] = 0; //<--------------------------
            }
        }

        __syncthreads();
         */

        // Each warp writes its part in shared memory
        if (laneId == WARP_SIZE - 1) {
            vmem[ threadIdx.x / WARP_SIZE ] = myCounter; // total found by this warp
        }


        // waits for all the warps to write
        __syncthreads();


        //Final reduction to get reduction over block, lets each warp do it redundantly
        myCounter = 0;
        if (laneId < nrWarpInBlock)
            myCounter = vmem[laneId];

        for (int i = WARP_SIZE / 2; i >= 1; i = i / 2) {
            myCounter += __shfl_xor(myCounter, i, WARP_SIZE);
        }

        toReturn = myCounter;

    } else {
        if (threadIdx.x == 0 && blockIdx.x == 0)
            printf("\nblockReduceFloat Failed %d !!\n", nrWarpInBlock);
    }
    return toReturn;
}




#ifdef RUNONGPU

__device__
#endif
int blockReduce(int myCounter, unsigned int WARP_SIZE) {



    int laneId = threadIdx.x & (WARP_SIZE - 1);
    int nrWarpInBlock = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    int toReturn = -1;

    if (nrWarpInBlock <= PHY_WRP_SZ) {
        //warp reduce
        for (int i = WARP_SIZE / 2; i >= 1; i = i / 2) {
            myCounter += __shfl_xor(myCounter, i, WARP_SIZE);
        }


        // Shared memory to gather result from all warps
        __shared__ int foundByWarps[ PHY_WRP_SZ ];
        volatile int* vmem = foundByWarps;

        /*
        //Initialize to zeros; only by one warp
        if ((threadIdx.x / WARP_SIZE) == 0) {

            for (int i = laneId; i < WARP_SIZE; i = i + WARP_SIZE) {
                vmem[i] = 0; //<--------------------------
            }
        }

        __syncthreads();
         */

        // Each warp writes its part in shared memory
        if (laneId == WARP_SIZE - 1) {
            vmem[ threadIdx.x / WARP_SIZE ] = myCounter; // total found by this warp
        }


        // waits for all the warps to write
        __syncthreads();


        //Final reduction to get reduction over block, lets each warp do it redundantly
        myCounter = 0;
        if (laneId < nrWarpInBlock)
            myCounter = vmem[laneId];

        for (int i = WARP_SIZE / 2; i >= 1; i = i / 2) {
            myCounter += __shfl_xor(myCounter, i, WARP_SIZE);
        }

        toReturn = myCounter;

    } else {
        if (threadIdx.x == 0 && blockIdx.x == 0)
            printf("\nblockReduce Failed %d !!\n", nrWarpInBlock);
    }
    return toReturn;
}


#ifdef RUNONGPU

__device__
#endif

float compute_weight_of_slef_loops_of_a_node(int node, int laneId, int nr_neighors, unsigned int* neighbors,
        float* weights_of_links_to_neighbors, int type, unsigned int WARP_SIZE) {



    float partial_weight_of_self_loop = 0.0;
    int shouldAdd = 0;
    for (int i = laneId; i < nr_neighors; i = i + WARP_SIZE) {

        shouldAdd = 0;

        if (neighbors[i] == (unsigned int) node) {

            shouldAdd = 1;
        }


        if (type == WEIGHTED) {

            if (shouldAdd)
                partial_weight_of_self_loop = partial_weight_of_self_loop + (float) shouldAdd * weights_of_links_to_neighbors[i];

        } else {

            partial_weight_of_self_loop = partial_weight_of_self_loop + (float) shouldAdd;
        }
    }


    // Reduction across warp afterwards will give the total_weighted_degee of 'node'

    for (int i = WARP_SIZE / 2; i >= 1; i = i / 2) {

        partial_weight_of_self_loop += __shfl_xor(partial_weight_of_self_loop, i, WARP_SIZE);
    }

    return partial_weight_of_self_loop;



}


#ifdef RUNONGPU

__device__
#endif
float comWDegOfNodeByBlock(int workerId, int nr_neighors, float* weightsOfLinks,
        int type, int nrWorker, unsigned int wrpSz) {

    if (type == UNWEIGHTED) {
        return (float) nr_neighors;
    } else {
        // each thread reads part of neighborhood
        float partialWdeg = 0.0;
        for (int i = workerId; i < nr_neighors; i = i + nrWorker) {
            partialWdeg = partialWdeg + weightsOfLinks[i];

        }
        // Reduction across warp afterwards will give the total_weighted_degee of 'node'

        partialWdeg = (float) blockReduce((int) partialWdeg, wrpSz);

        return partialWdeg;

    }

}
#ifdef RUNONGPU

__device__
#endif
float comWDegOfNode(int laneId, int nr_neighors, float* weights_of_links_to_neighbors, int type, unsigned int WARP_SIZE) {

    if (type == UNWEIGHTED) {
        return (float) nr_neighors;
    } else {
        // each thread reads part of neighborhood
        float partialWdeg = 0.0;
        for (int i = laneId; i < nr_neighors; i = i + WARP_SIZE) {
            partialWdeg = partialWdeg + weights_of_links_to_neighbors[i];

        }
        // Reduction across warp afterwards will give the total_weighted_degee of 'node'

        for (int i = WARP_SIZE / 2; i >= 1; i = i / 2) {

            partialWdeg += __shfl_xor(partialWdeg, i, WARP_SIZE);
        }

        return partialWdeg;

    }

}


#ifdef RUNONGPU

__global__
#endif
void preComputeWdegs(int *indices, float* weights, float *wDegs, int type, unsigned int nrComms, int WARP_SIZE) {

    unsigned int wid = threadIdx.x / WARP_SIZE;
    unsigned int laneId = threadIdx.x % WARP_SIZE; // id in the warp

    // Global warp ID and pointer in global Memory
    wid = blockIdx.x * (blockDim.x / WARP_SIZE) + wid;
    while (wid < nrComms) {

        unsigned int startNbr = indices[wid];
        unsigned int endNbr = indices[wid + 1];

        float wdeg = endNbr - startNbr;

        if (type == WEIGHTED) {
            wdeg = comWDegOfNode(laneId, (endNbr - startNbr), &weights[startNbr], type, WARP_SIZE);
        }

        wDegs[wid] = wdeg;

        wid = wid + (blockDim.x * gridDim.x) / WARP_SIZE;

    }

}


#ifdef RUNONGPU

__device__
#endif
int hashInsertGPU(HashItem* Table, unsigned int* totNrAttempt,
        unsigned int bucketSize, HashItem *dataItem, float* tot, float wDegNode,
        float m2, float* bestGain, int *bestDest, int sCId) {

    //unsigned int wid = threadIdx.x / WARP_SIZE;
    //unsigned int laneId = threadIdx.x % WARP_SIZE; // id in the warp
    float addedValue = 0.0, prevValue = 0.0;
    unsigned int i = 0, j = 0;

    unsigned int h1 = H1GPU(dataItem->cId, bucketSize); // h1


    unsigned int h2 = H2GPU(dataItem->cId, bucketSize);
    do {

        j = (h1 + i * h2) % bucketSize;

        int currCId = atomicCAS((int*) &Table[j].cId, FLAG_FREE, (1 + dataItem->cId)); //NOTE: HashTable stores (cid+1)

        // the winner might be sleeping  while the losers might run and succeed else if () test !!!!!!!!!

        if (currCId == FLAG_FREE) { // new cId @ location j;  exactly ONE winner

            addedValue = dataItem->gravity;

            prevValue = atomicAdd((float*) &Table[j].gravity, addedValue);

            //if (prevValue > 0.0)printf("\nUnexpected value in HashTable\n");

            prevValue = prevValue + addedValue;

            //float gain = (prevValue * m2 - tot[dataItem->cId] * wDegNode);

            double dgain = 0.0;
            if (dataItem->cId != sCId)
                //dgain =  2.0 *  prevValue - 2.0 *  wDegNode * (tot[dataItem->cId] -  tot[sCId] +  wDegNode)*  (1.0 / (double) m2);
                dgain = (double) (2.0 * (double) prevValue - 2.0 * (double) wDegNode * ((double) tot[dataItem->cId] - (double) tot[sCId] + (double) wDegNode)* (1.0 / (double) m2));

            float gain = (float) dgain;

            //if(dataItem->cId == 97)
            //printf("\npcId= %d gpc= %f tot[pc]= %f wDeg= %f sCId= %d tot[sCId]= %f m2= %f gain= %f\n",dataItem->cId, prevValue, tot[dataItem->cId], wDegNode, sCId, tot[sCId], m2, gain);

            if ((gain > *bestGain) || (gain == *bestGain && gain != 0 && dataItem->cId < *bestDest)) {

                *bestGain = gain;
                *bestDest = dataItem->cId;
            }

            //*totNrAttempt = *totNrAttempt + (i + 1);
            return (int) i;

        } else if (currCId == (1 + dataItem->cId)) { //existing cId; what if multiple losers came with same currCId !!!!!!!!!!!!!!!

            addedValue = dataItem->gravity;

            prevValue = atomicAdd((float*) &Table[j].gravity, addedValue); //Table[j].gravity += dataItem->gravity;

            prevValue = prevValue + addedValue;

            //float gain = (prevValue * m2 - tot[dataItem->cId] * wDegNode);

            // double  dgain= (double)(2.0* (double)prevValue  - 2.0*(double)tot[dataItem->cId] * (double)wDegNode * (double)(1.0/(double)m2));
            double dgain = 0.0;
            if (dataItem->cId != sCId)
                dgain = (double) (2.0 * (double) prevValue - 2.0 * (double) wDegNode * ((double) tot[dataItem->cId] - (double) tot[sCId] + (double) wDegNode)* (double) (1.0 / (double) m2));

            float gain = (float) dgain;

            //if(dataItem->cId == 97)
            //printf("\npcId= %d gpc= %f tot[pc]= %f wDeg= %f sCId= %d tot[sCId]= %f m2= %f\n",dataItem->cId, prevValue, tot[dataItem->cId], wDegNode, sCId, tot[sCId], m2);

            if ((gain > *bestGain) || (gain == *bestGain && gain != 0 && dataItem->cId < *bestDest)) {

                *bestGain = gain;
                *bestDest = dataItem->cId;
            }
            //*totNrAttempt = *totNrAttempt + (i + 1);
            return (int) i;

        } else {
            i = i + 1;
        }

    } while (i < bucketSize);
    return -1;
}

#ifdef RUNONGPU

__device__
#endif
int hashSearchGPU(HashItem* Table, unsigned int* totNrAttempt, unsigned int bucketSize, HashItem *dataItem) {

    unsigned int i = 0, j = 0;

    unsigned int h1 = H1GPU(dataItem->cId, bucketSize); // h1
    unsigned int h2 = H2GPU(dataItem->cId, bucketSize);

    do {

        j = (h1 + i * h2) % bucketSize;

        if (Table[j].cId == (1 + dataItem->cId)) {
            *totNrAttempt = *totNrAttempt + (i + 1);
            return (int) j; // returning the index where the  key is found
        } else {
            i = i + 1;
        }
    } while (i < bucketSize && Table[j].cId != FLAG_FREE);
    return -1;
}
#ifdef RUNONGPU

__device__
#endif
int hashSearchModified(HashItem* Table, unsigned int* totNrAttempt, unsigned int bucketSize, HashItem *dataItem) {

    unsigned int i = 0, j = 0;

    unsigned int h1 = H1GPU(dataItem->cId, bucketSize); // h1
    unsigned int h2 = H2GPU(dataItem->cId, bucketSize);

    do {

        j = (h1 + i * h2) % bucketSize;

        if (Table[j].cId == (1 + dataItem->cId)) {
            *totNrAttempt = *totNrAttempt + (i + 1);
            return (int) j; // returning the index where the  key is found
        } else {
            i = i + 1;
        }
    } while (i < bucketSize /*&& Table[j].cId != FLAG_FREE*/);
    return -1;
}

#ifdef RUNONGPU

__device__
#endif
void intraWarpBest(int &bestDestination, float &bestGain, unsigned int WARP_SIZE) {

    for (int i = WARP_SIZE / 2; i >= 1; i = i / 2) {

        float recvGain = __shfl_xor(bestGain, i, WARP_SIZE);
        int recvDest = __shfl_xor(bestDestination, i, WARP_SIZE);

        if ((recvGain > bestGain) || (recvGain == bestGain && recvDest < bestDestination)) {

            bestGain = recvGain;
            bestDestination = recvDest;
        }
    }
}
#ifdef RUNONGPU

__device__
#endif
int findPrimebyWarp(int* primes, int nrPrime, int threshold, unsigned int WARP_SIZE) {



    unsigned int laneId = threadIdx.x & (WARP_SIZE - 1);
    int myPrime = primes[nrPrime - 1]; // Largest one in the file as MAXPRIME

    if (threshold > myPrime)
        return -1;

    for (unsigned int i = laneId; i < nrPrime; i = i + WARP_SIZE) {

        int currentPrime = __ldg(&primes[i]);

        if (currentPrime > threshold) {
            myPrime = currentPrime;
            break;
        }
    }
    for (unsigned int i = WARP_SIZE / 2; i >= 1; i = i / 2) {
        int received = __shfl_xor(myPrime, i, WARP_SIZE);
        if (received < myPrime)
            myPrime = received;
    }
    return myPrime;

}


#ifdef RUNONGPU

__global__
#endif
void preComputePrimes(int *primes, int nrPrimes, unsigned int* thresholds, int nrBigBlock, int *selectedPrimes, int WARP_SIZE) {

    unsigned int wid = threadIdx.x / WARP_SIZE;
    unsigned int laneId = threadIdx.x & (WARP_SIZE - 1); // id in the warp

    // Global warp ID and pointer in global Memory
    wid = blockIdx.x * (blockDim.x / WARP_SIZE) + wid;

    while (wid < nrBigBlock) {

        int threshold = thresholds[wid];

        int myPrime = findPrimebyWarp(primes, nrPrimes, threshold, WARP_SIZE);

        selectedPrimes[wid] = myPrime;

        wid = wid + (blockDim.x * gridDim.x) / WARP_SIZE;

    }

}


#ifdef RUNONGPU

__device__
#endif
HashItem findTheBest(int bestDestination, float bestGain, unsigned int WARP_SIZE) {



    int laneId = threadIdx.x & (WARP_SIZE - 1);
    int nrWarpInBlock = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    HashItem toReturn;
    toReturn.cId = -1;
    toReturn.gravity = 0.0;

    if (nrWarpInBlock <= PHY_WRP_SZ) { // lets stick to it

        //Butterfly reduction to determine the best destination (intra-warp)

        intraWarpBest(bestDestination, bestGain, WARP_SIZE);

        // Shared memory to gather result from all warps
        __shared__ HashItem foundByWarps[ PHY_WRP_SZ ];
        volatile HashItem* vmem = foundByWarps;

        // Each warp writes its part in shared memory
        if (laneId == WARP_SIZE - 1) {
            vmem[ threadIdx.x / WARP_SIZE ].cId = bestDestination;
            vmem[threadIdx.x / WARP_SIZE].gravity = bestGain;
        }


        // waits for all the warps to write
        __syncthreads();

        //Butterfly reduction to determine the best destination (inter-warp)

        bestDestination = -1;
        bestGain = 0.0;

        if (laneId < nrWarpInBlock) {
            bestDestination = vmem[laneId].cId;
            bestGain = vmem[laneId].gravity;
        }

        intraWarpBest(bestDestination, bestGain, WARP_SIZE);

        toReturn.cId = bestDestination;
        toReturn.gravity = bestGain;

    } else {
        if (threadIdx.x == 0 && blockIdx.x == 0)
            printf("\nblockReduce Failed  nrWarpInBlock= %d !!\n", nrWarpInBlock);
    }
    return toReturn;
}
#ifdef RUNONGPU

__device__
#endif

void decideBestDest(int node, int workerId, int nr_neighbor,
        unsigned int* neighbors, float* weightsToNeighbors, int *n2c,
        float *in, float* tot, float wDegOfNode,
        float total_weight, int *nr_moves, float* tot_new, int* n2c_new,
        HashItem* shashTable, unsigned int bucketSize, int nrWorker,
        unsigned int wrpSz, int* cardinalityOfComms_old,
        int* cardinalityOfComms_new) {


    HashItem dataItem;
    unsigned int nrAttempts = 0;

    float bestGain = 0.0;
    int bestDestination = -1;
    int flagInsert = 0;
    float selfLoop = 0.0;
    /*
    if (workerId == 0 && node == EID)
        printf("\n---------------------------> Called decideBest-------\n");
     */
    for (int j = workerId; j <= nr_neighbor; j = j + nrWorker) {

        //if (node == 35)printf("laneId= %d,j= %d neighbor=%u nr_neighbor=%d \n", laneId, j, neighbors[j], nr_neighbor);

        if (j < nr_neighbor) {

            dataItem.cId = n2c[neighbors[j]];
            dataItem.gravity = (weightsToNeighbors == NULL) ? 1.0 : weightsToNeighbors[j];

            flagInsert = (node != (int) neighbors[j]); // NOTE: Ignore self-loop to "node"
            selfLoop += (!flagInsert)*(dataItem.gravity);
            //-----------Computation of Internals-----------------------//

            /*if (n2c[node] == dataItem.cId) {
                float internal = (weightsToNeighbors == NULL) ? 1.0 : weightsToNeighbors[j];
                atomicAdd(&in[dataItem.cId], internal);
            }*/

        } else if (j == nr_neighbor) {

            dataItem.cId = n2c[node];
            dataItem.gravity = 0.0;
            flagInsert = 1;
        }
        flagInsert = 1; // -------------------------------------------------------<<<<<<<NOTE IT
        if (flagInsert) { // ignore self-loop 

            //if(node==35)printf(" node= %d laneId= %d,dataItem.cId= %d gravity=%f \n", node, laneId, dataItem.cId , dataItem.gravity);

            hashInsertGPU(shashTable, &nrAttempts, bucketSize, &dataItem, tot,
                    wDegOfNode, total_weight, &bestGain, &bestDestination, n2c[node]);

        }
    }

    __syncthreads(); // MUST NEEDED ?

    selfLoop = blockReduceFloat(selfLoop, wrpSz);

    HashItem sourceItem;
    sourceItem.cId = n2c[node];
    flagInsert = hashSearchGPU(shashTable, &nrAttempts, bucketSize, &sourceItem);
    if (flagInsert >= 0) {

        sourceItem = shashTable[flagInsert];

        sourceItem.cId = sourceItem.cId - 1; //NOTE: inserted as (sourceItem.cId+1)--------------------<<

        bestGain = bestGain - 2.0 * sourceItem.gravity + 2.0 * selfLoop;

    } else {
        printf("\nEveryone must find sourceItem.cId*\n");
    }


    if (!workerId)atomicAdd(&in[node], sourceItem.gravity); // Only one thread should do it

    HashItem bestItem = findTheBest(bestDestination, bestGain, wrpSz);

    bestDestination = bestItem.cId;
    bestGain = bestItem.gravity;

    //thread that processed the position "nr_neighbor" still has n2c[node] in dataItem.cId
    int lastWorker = nr_neighbor % nrWorker;
    if (workerId == lastWorker) {

        //if(node==EID)
        //printf("\n--------------------->node= %d bestDest= %d bestGain= %f\n", node, bestDestination, bestGain);

        //printf("landId = %d, dataItem.cId= %d node=%d, nr_neighbor=%d\n", laneId, dataItem.cId, node, nr_neighbor);

        if (dataItem.cId != n2c[node])
            printf("Impossible; Something is wrong; node= %d workerId=%d", node, workerId);

        if (bestDestination >= 0 && bestDestination != dataItem.cId && bestGain > 0) {


            if (cardinalityOfComms_old[ dataItem.cId] == 1 && cardinalityOfComms_old[bestDestination] == 1 && bestDestination > dataItem.cId) {
                bestDestination = dataItem.cId;
                //printf("protecting %d  bestD= %d d.cId = %d \n", node, bestDestination, dataItem.cId); 
            } else {

                (*nr_moves)++;


                atomicAdd(&tot_new[dataItem.cId], (-1) * wDegOfNode);
                atomicAdd(&cardinalityOfComms_new[dataItem.cId], -1);
                atomicAdd(&tot_new[bestDestination], wDegOfNode);
                atomicAdd(&cardinalityOfComms_new[bestDestination], 1);

                n2c_new[node] = bestDestination;
            }

        } else {
            n2c_new[node] = n2c[node];
            //printf("node  %d, not moving\n", node);
        }
    }
}
#ifdef RUNONGPU

__device__
#endif

void compute_neighboring_communites_using_Hash(int node, int laneId,
        int nr_neighbor, unsigned int* neighbors, float* weightsToNbors,
        int *n2c, float *in, float* tot, float weighted_degree_of_node,
        float total_weight, int *nr_moves, float* tot_new, int* n2c_new,
        HashItem* shashTable, unsigned int bucketSize,
        int* cardinalityOfComms_old, int* cardinalityOfComms_new,
        unsigned int WARP_SIZE) {




    HashItem dataItem;
    unsigned int nrAttempts = 0;

    float bestGain = 0.0;
    int bestDestination = -1;

    int flagInsert = 0;
    float selfLoop = 0.0;

    // hash all neighbors of current node and community of current node (when j=nr_neighbor)
    //if (!laneId && (node==97||node==2 )) printf("node= %d nr_neighbor=%d \n", node, nr_neighbor);

    for (int j = laneId; j <= nr_neighbor; j = j + WARP_SIZE) {

        //if (node == 35)printf("laneId= %d,j= %d neighbor=%u nr_neighbor=%d \n", laneId, j, neighbors[j], nr_neighbor);

        if (j < nr_neighbor) {

            dataItem.cId = n2c[neighbors[j]];
            dataItem.gravity = (weightsToNbors == NULL) ? 1.0 : weightsToNbors[j];

            flagInsert = (node != (int) neighbors[j]); // NOTE: Ignore self-loop to "node"
            selfLoop += (!flagInsert) * dataItem.gravity;
            //-----------Computation of Internals-----------------------//

            /*if (n2c[node] == dataItem.cId) {
                float internal = (weightsToNbors == NULL) ? 1.0 : weightsToNbors[j];
                atomicAdd(&in[dataItem.cId], internal);
            }*/

        } else if (j == nr_neighbor) {

            dataItem.cId = n2c[node];
            dataItem.gravity = 0.0;
            flagInsert = 1;
        }
        flagInsert = 1; // --------------------------------------------------<<<<<<<<<<<< NOTE THIS
        if (flagInsert) { // ignore self-loop 

            // if (DUMP) if (node == 14)printf(" node= %d laneId= %d,dataItem.cId= %d g = %f tot[%d] = %f \n", node, laneId, dataItem.cId, dataItem.gravity, dataItem.cId, tot[dataItem.cId]);

            hashInsertGPU(shashTable, &nrAttempts, bucketSize, &dataItem, tot,
                    weighted_degree_of_node, total_weight, &bestGain, &bestDestination, n2c[node]);

        }
    }

    //if (DUMP && node == 8)printf(" node = %d, laneId= %d , bestGain =%f wDeg= %f  bestDest= %d  n2c[node]= %d\n", node, laneId, bestGain, weighted_degree_of_node, bestDestination, n2c[node]);

    //Butterfly reduction to determine best destination
    HashItem sourceItem;
    sourceItem.cId = n2c[node];
    sourceItem.gravity = -5.0;
    flagInsert = hashSearchGPU(shashTable, &nrAttempts, bucketSize, &sourceItem);
    if (flagInsert >= 0) {

        sourceItem = shashTable[flagInsert];

        sourceItem.cId = sourceItem.cId - 1; //NOTE: inserted as (sourceItem.cId+1)--------------------<<

        //if (DUMP && node == 8)printf("\n sourceItem.cId= %d, sourceItem.gravity =%f \n", sourceItem.cId, sourceItem.gravity);
        // ---> bestGain  = bestGain -  2.0 * sourceItem.gravity; 

    } else {
        printf("\nEveryone must find sourceItem.cId\n");
    }


    if (!laneId)atomicAdd(&in[node], sourceItem.gravity); // Only one thread should do it

    //if ( !laneId)printf("\n node = %d sourceItem.cId= %d, sourceItem.gravity =%f \n",  node, sourceItem.cId, sourceItem.gravity);
    for (int i = WARP_SIZE / 2; i >= 1; i = i / 2) {

        selfLoop += __shfl_xor(selfLoop, i, WARP_SIZE);
    }

    //if(!laneId)printf("\n%d : %f sg\n", node, sourceItem.gravity);

    //----> bestGain = bestGain -2.0* selfLoop;

    for (int i = WARP_SIZE / 2; i >= 1; i = i / 2) {

        float recvGain = __shfl_xor(bestGain, i, WARP_SIZE);
        int recvDest = __shfl_xor(bestDestination, i, WARP_SIZE);

        if ((recvGain > bestGain) || (recvGain == bestGain && recvDest < bestDestination)) {

            bestGain = recvGain;
            bestDestination = recvDest;

        }
    }

    //if(node == 2) printf("\n%d %f %f %f %d pqr\n", node, bestGain, sourceItem.gravity, selfLoop, bestDestination);

    //if( bestGain < ( 2.0*sourceItem.gravity + 2.0*selfLoop))
    //printf("\nCHANGE IN DECISION\n");

    bestGain = bestGain - 2.0 * sourceItem.gravity + 2.0 * selfLoop;

    //if (DUMP && node == 8)printf(" node = %d, laneId= %d , sourceItem.gravity= %f selfLoop =%f \n", node, laneId, sourceItem.gravity, selfLoop);
    /*
    // If bestDest is to determined after searching the Hash

    float bestGain2 = bestGain;
    int bestDestination2 = bestDestination;

    bestGain = 0.0;
    bestDestination = -1;

    for (int j = laneId; j <= nr_neighbor; j = j + WARP_SIZE) {

        if (j < nr_neighbor) {
            dataItem.cId = n2c[neighbors[j]];
        } else if (j == nr_neighbor) {
            dataItem.cId = n2c[node];
        }

        // use flagInsert to store the index of search result

        flagInsert = -1;

        flagInsert = hashSearchGPU(shashTable, &nrAttempts, bucketSize, &dataItem);

        if (flagInsert < 0) {

            printf("\nWhile searching dataItem.cId= %d, laneId= %d flagInsert =%d \n", dataItem.cId, laneId, flagInsert);
        }

        if (flagInsert >= 0) {

            dataItem = shashTable[flagInsert];

            dataItem.cId = dataItem.cId - 1; //NOTE: inserted as (dataItem.cId+1)--------------------<<

            float gain = dataItem.gravity; //- (tot[dataItem.cId] * weighted_degree_of_node) / total_weight;

            //if (node == 18459128)printf("\nSearch: laneId= %d,dataItem.cId= %d gravity=%f gain= %f \n", laneId, dataItem.cId, dataItem.gravity, gain);

            if ((gain > bestGain) || (gain == bestGain && dataItem.cId < bestDestination)) {

                bestGain = gain;
                bestDestination = dataItem.cId;
            }
        }
    }


    //Butterfly reduction to determine best destination

    for (int i = WARP_SIZE / 2; i >= 1; i = i / 2) {

        float recvGain = __shfl_xor(bestGain, i, WARP_SIZE);
        int recvDest = __shfl_xor(bestDestination, i, WARP_SIZE);

        if ((recvGain > bestGain) || (recvGain == bestGain && recvDest < bestDestination)) {
            bestGain = recvGain;
            bestDestination = recvDest;
        }
    }

    if ((bestDestination != bestDestination2) && (laneId <= nr_neighbor)) {
        printf("laneId= %d node=%d, bestDest=%d bestDest2=%d bestGain= %f bestGain2=%f \n",
                laneId, node, bestDestination, bestDestination2, bestGain, bestGain2);
    }

     */
    //if ((node == 97 || node == 2) && laneId % (WARP_SIZE) == 0)
    //printf(" node= %d  bestDest= %d bestGain = %f \n", node, bestDestination, bestGain);


    //thread that processed the position "nr_neighbor" still has n2c[node] in dataItem.cId

    int sourceLane = nr_neighbor % WARP_SIZE;
    if (laneId == sourceLane) {
        /*
        if (node == 8 && DUMP)
            printf("-------------> landId = %d, dataItem.cId= %d node=%d, nr_neighbor=%d dest= %d Gain= %f\n", laneId, dataItem.cId, node, nr_neighbor, bestDestination, bestGain);
         */
        if (dataItem.cId != n2c[node])
            printf("Impossible; Something is wrong; node= %d laneId=%d", node, laneId);

        if (bestDestination >= 0 && bestDestination != dataItem.cId && bestGain > 0) {


            if (cardinalityOfComms_old[ dataItem.cId] == 1 && cardinalityOfComms_old[bestDestination] == 1 && bestDestination > dataItem.cId) {
                //printf("*protecting %d  bestD= %d d.cId = %d \n", node, bestDestination, dataItem.cId); 
                bestDestination = dataItem.cId;
            } else {

                (*nr_moves)++;


                atomicAdd(&tot_new[dataItem.cId], (-1) * weighted_degree_of_node);
                atomicAdd(&cardinalityOfComms_new[dataItem.cId], -1);
                atomicAdd(&tot_new[bestDestination], weighted_degree_of_node);
                atomicAdd(&cardinalityOfComms_new[bestDestination], 1);

                n2c_new[node] = bestDestination;
            }
        } else {
            n2c_new[node] = n2c[node];
            // printf("node  %d, not moving\n", node);
        }

    }

    *nr_moves = __shfl(*nr_moves, sourceLane, WARP_SIZE);

    for (int j = laneId; j < bucketSize; j = j + WARP_SIZE) {
        shashTable[j].cId = 0;
        shashTable[j].gravity = 0.0;
    }
}

#ifdef RUNONGPU

__global__
#endif

void neigh_comm(int community_size, int* indices, unsigned int* links,
        float* weights, int *n2c, float *in, float* tot, int type, int *n2c_new,
        float* tot_new, int* movement_record, double total_weight,
        unsigned int bucketSzLimit, int* candidateComms, int nrCandidate,
        int* primes, int nrPrime, int* cardinalityOfComms_old,
        int* cardinalityOfComms_new, unsigned int WARP_SIZE, float *wDegs) {


    unsigned int wid = threadIdx.x / WARP_SIZE;
    unsigned int laneId = threadIdx.x % WARP_SIZE; // id in the warp


    extern __shared__ HashItem __sMem[];
    HashItem* shashTable = __sMem + wid * bucketSzLimit; // NOTE: bucketSize == size of Table in "Warp Memory"

    // Warp based initialization

    for (unsigned int i = laneId; i < bucketSzLimit; i = i + WARP_SIZE) {
        shashTable[i].cId = 0;
        shashTable[i].gravity = 0.0;
    }


    // Global warp ID
    wid = blockIdx.x * (blockDim.x / WARP_SIZE) + wid;

    unsigned int cId = wid;
    int nr_moves = 0;

    while (cId < nrCandidate) {

        int node = candidateComms[cId];

        int startOfNhood = indices[node];
        int endOfNhood = indices[node + 1];

        int nr_neighbor = endOfNhood - startOfNhood;

        int activeBktSz = bucketSzLimit;
        /*
        if (nr_neighbor > (bucketSzLimit / 2))
            activeBktSz = bucketSzLimit;
        else
            activeBktSz = findPrimebyWarp(primes, nrPrime, (nr_neighbor * 7) / 4);

        if (activeBktSz < 0 || activeBktSz > bucketSzLimit) {

            activeBktSz = findPrimebyWarp(primes, nrPrime, nr_neighbor);
            if (!laneId)
                printf("\n-----------Nearest Prime Can't be negative or > bktSzLimit-- szNhd=%d \n", nr_neighbor);
        }
         */

        //if (nr_neighbor <= (bucketSize * LOAD_FACTOR) / 3) {

        float *weightsMem = NULL;
        if (type == WEIGHTED) {
            weightsMem = &weights[startOfNhood];
        }

        float wdegNode = wDegs[node];
        /*
        float wdegNode = 0.0;
 
        if (type == WEIGHTED)
                wdegNode = comWDegOfNode(laneId, nr_neighbor, weightsMem, type, WARP_SIZE);

        else
                wdegNode = (float)nr_neighbor;

        if(wDegs[node] != wdegNode)
                if(!laneId) printf("\n wDegs PROBLEM neigh_comm\n");
         */
        compute_neighboring_communites_using_Hash(node, laneId, nr_neighbor,
                &links[startOfNhood], weightsMem, n2c, in, tot, wdegNode,
                total_weight, &nr_moves, tot_new, n2c_new, shashTable,
                activeBktSz, cardinalityOfComms_old, cardinalityOfComms_new,
                WARP_SIZE);
        // }

        cId = cId + (blockDim.x * gridDim.x) / WARP_SIZE;
    }
    /*
    if (!laneId)
        movement_record[wid] = nr_moves;

    if (!laneId && !wid)
        printf("\nReturning From Kernel\n");
     */

}

#ifdef RUNONGPU

__global__
#endif
void lookAtNeigboringComms(int* indices, unsigned int* links, float* weights,
        int *n2c, float *in, float* tot, int type, int *n2c_new, float *in_new,
        float* tot_new, int* movement_record, double total_weight,
        int* candidateComms, int nrCandidateComms, HashItem* gblTable,
        int* glbTblPtrs, int* primes, int nrPrime, unsigned int wrpSz,
        int* cardinalityOfComms_old, int* cardinalityOfComms_new, float *wDegs) {

    HashItem* blockTable = NULL;
    float *weightsMem = NULL;

    __shared__ HashItem blkTblShared[SHARED_TABLE_SIZE];

    //NOTE: Make sure host module has allocated at least 2 times memory than upper limit
    //for global Hash Table

    int tblStart = 2 * glbTblPtrs [blockIdx.x];
    blockTable = &gblTable[tblStart ];
    int nr_moves = 0;
    unsigned int bucketSize;

    unsigned int commIndex = blockIdx.x;

    while (commIndex < nrCandidateComms) {

        int node = candidateComms[commIndex];

        //if(threadIdx.x==0)
        //printf("nodeID= %d\n", node);

        unsigned int startOfNhd = indices[node];
        unsigned int endOfNhd = indices[node + 1];

        int nr_neighbor = (endOfNhd - startOfNhd);


        if (nr_neighbor < (SHARED_TABLE_SIZE * CAPACITY_FACTOR_NUMERATOR / CAPACITY_FACTOR_DENOMINATOR)) {

            blockTable = blkTblShared;
            bucketSize = SHARED_TABLE_SIZE;

            /*	
            int nearestPrime = findPrimebyWarp(primes, nrPrime, (nr_neighbor * 3) / 2);
            if (nearestPrime < 0 || nearestPrime > SHARED_TABLE_SIZE)
                nearestPrime = findPrimebyWarp(primes, nrPrime, nr_neighbor);

            bucketSize = nearestPrime;

            if ((nearestPrime < 0 || nearestPrime > SHARED_TABLE_SIZE) && (threadIdx.x == 0 || threadIdx.x == blockDim.x - 1))
                printf("\n-----------Nearest Prime Can't be negative or > SHARED_TS-----------maxSzNhd=%d \n", nr_neighbor);
	
             */

        } else {

            blockTable = &gblTable[tblStart];

            // choose prime number > nr_neighbor
            int nearestPrime = findPrimebyWarp(primes, nrPrime, (nr_neighbor * 3) / 2, wrpSz);

            if (nearestPrime < 0)
                nearestPrime = findPrimebyWarp(primes, nrPrime, nr_neighbor, wrpSz);

            if (nearestPrime < 0 && (threadIdx.x == 0 || threadIdx.x == blockDim.x - 1))
                printf("\n-----------Nearest Prime Can't be negative-----------maxSzNhd=%d \n", nr_neighbor);

            bucketSize = nearestPrime;
        }

        if (type == WEIGHTED) {
            weightsMem = &weights[startOfNhd];
        }

        // Clear HashTable
        initByBlock(blockTable, bucketSize, threadIdx.x, blockDim.x);
        float wDegNode = wDegs[node];
        /*
        // Lets get it done by warp; each warp will do the same independently
        float wDegNode = 0.0;
        //if (nr_neighbor <= 480)
        if (type == WEIGHTED)
                wDegNode = comWDegOfNode(threadIdx.x & (wrpSz - 1), nr_neighbor, weightsMem, type, wrpSz);
        else	
                wDegNode = (float)nr_neighbor; 
        //else
        //wDegNode = comWDegOfNodeByBlock(threadIdx.x, nr_neighbor, weightsMem, type, blockDim.x);
        if(wDegs[node] != wDegNode)
                if(threadIdx.x == 0)printf("\nwDegs PROBLEM lookAtNeighComm\n");

        if (threadIdx.x == 0 && node == EID)
            printf("\n------> Before Call to decideBestDest\n");
         */
        decideBestDest(node, threadIdx.x, nr_neighbor, &links[startOfNhd],
                weightsMem, n2c, in, tot, wDegNode, total_weight,
                &nr_moves, tot_new, n2c_new, blockTable,
                bucketSize, blockDim.x, wrpSz, cardinalityOfComms_old,
                cardinalityOfComms_new);


        commIndex += gridDim.x;

    }

    nr_moves = blockReduce(nr_moves, wrpSz);
    // Only one thread in a block will do the  following
    if (!threadIdx.x) {
        //---> movement_record[blockIdx.x] = nr_moves;
        if (nr_moves < 0)
            printf("\n nr_moves can't be negative \n");
    }
    __syncthreads(); // Do you really need it ?
    /*
    if (!threadIdx.x && !blockIdx.x)
        printf("\nReturning From Kernel\n");
    */
}

/**
 * Returns bucketSize if inserted new record in the table else position of record 
 * @param Table
 * @param bucketSize
 * @param dataItem
 * @return 
 */
#ifdef RUNONGPU

__device__
#endif
int hashInsertSimple(HashItem* Table, unsigned int bucketSize, HashItem *dataItem) {


    unsigned int i = 0, j = 0;
    unsigned int h1 = H1GPU(dataItem->cId, bucketSize); // h1
    unsigned int h2 = H2GPU(dataItem->cId, bucketSize);


    do {

        j = (h1 + i * h2) % bucketSize;

        //NOTE: HashTable stores (cid+1)
        int currCId = atomicCAS((int*) &Table[j].cId, FLAG_FREE, (1 + dataItem->cId));

        if (currCId == FLAG_FREE) { // new cId @ location j;  exactly ONE winner

            //printf("\nCame with %d \n", dataItem->cId);

            atomicAdd((float*) &Table[j].gravity, dataItem->gravity);
            //return (int) i;
            return (int) bucketSize; // returns bucketSize to indicate new in table

        } else if (currCId == (1 + dataItem->cId)) {

            atomicAdd((float*) &Table[j].gravity, dataItem->gravity);
            return (int) i;

        } else {
            i = i + 1;
        }

    } while (i < bucketSize);

    return -1;
}
#ifdef RUNONGPU

__device__
#endif
int hashInsertSimple2(HashItem* Table, unsigned int bucketSize, HashItem *dataItem, int cId, int node) {


    unsigned int i = 0, j = 0;
    unsigned int h1 = H1GPU(dataItem->cId, bucketSize); // h1
    unsigned int h2 = H2GPU(dataItem->cId, bucketSize);


    do {

        j = (h1 + i * h2) % bucketSize;

        //NOTE: HashTable stores (cid+1)
        int currCId = atomicCAS((int*) &Table[j].cId, FLAG_FREE, (1 + dataItem->cId));

        //if (cId == CID && currCId == FLAG_FREE )
        //printf("\nj = %u Trying %d but found %d tid= %d h1=%u h2= %u node= %d\n", j, dataItem->cId, currCId, threadIdx.x, h1, h2, node);

        if (currCId == FLAG_FREE) { // new cId @ location j;  exactly ONE winner

            //if(cId==CID)printf("\nCame with %d  threadid.x = %u node = %d bucketSize=%u cid = %d j = %u\n", dataItem->cId, threadIdx.x, node, bucketSize, cId, j);

            atomicAdd((float*) &Table[j].gravity, dataItem->gravity);
            //return (int) i;
            return (int) bucketSize; // returns bucketSize to indicate new in table

        } else if (currCId == (1 + dataItem->cId)) {

            atomicAdd((float*) &Table[j].gravity, dataItem->gravity);
            return (int) i;

        } else {
            i = i + 1;
        }

    } while (i < bucketSize);

    printf("\n Can't Happen  for neighbor= %d of node= %d bucketSize= %d \n", dataItem->cId, cId, bucketSize);

    return -1;
}
#ifdef RUNONGPU

__device__
#endif
void initWarpHashTable(HashItem* shashTable, unsigned int bucketSize,
        unsigned int laneId, unsigned int WARP_SIZE) {

    // Warp based initialization
    for (unsigned int i = laneId; i < bucketSize; i = i + WARP_SIZE) {
        shashTable[i].cId = FLAG_FREE;
        shashTable[i].gravity = 0.0;
    }
}

#ifdef RUNONGPU

__device__
#endif
void hashNeighborsOfNode(unsigned int cid, HashItem* hashTable, float* weightsOfNhood,
        int szNhood, unsigned int* neighbors, int* n2c, int* renumber,
        unsigned int bucketSize, unsigned int laneId, int *nrDiscoveredByMe,
        unsigned int WARP_SIZE) {

    unsigned int wid = threadIdx.x / WARP_SIZE;
    wid = blockIdx.x * (blockDim.x / WARP_SIZE) + wid;
    HashItem item;
    for (unsigned int i = laneId; i < szNhood; i = i + WARP_SIZE) {

        item.cId = renumber[n2c[neighbors[i]]];
        item.gravity = (weightsOfNhood == NULL) ? 1.0 : weightsOfNhood[i];

        //hash the item into hasTable
        int retValue = hashInsertSimple(hashTable, bucketSize, &item);

        /*
        if (DUMP && cid == 6)
            printf("\n-----------wid = %u laneId=%u  Hasing %d nId= %u retValue=%d  bucketSize = %u \n", wid, laneId, item.cId, neighbors[i], retValue, bucketSize);
         */

        //returned bucketSize if inserted new record else position of record 
        if (retValue == bucketSize)
            *nrDiscoveredByMe = *nrDiscoveredByMe + 1;

        //Negate if it was inserted as new in the table
        retValue = (retValue == bucketSize) ? -1 : 1;
        neighbors[i] = (1 + neighbors[i]) * retValue; // nId is now (nId+1) OR -(nId+1)

        /*
        if (wid == 1)
            printf("\n-----**------wid = %u laneId=%u  Hasing %d nId= %u retValue=%d \n", wid, laneId, item.cId, neighbors[i], retValue);
         */
    }

}

/**
 * 
 * @param hashTable
 * @param szNhood
 * @param neighbors
 * @param n2c
 * @param renumber
 * @param bucketSize
 * @param laneId
 * @param offsetInGlobalMem points where each thread should be writing for the NewComm ;
 * @param newLinksOfNewComm
 * @param newWeightsOfNewComm
 */
#ifdef RUNONGPU

__device__
#endif
void collectFromHashTable(unsigned int cid, HashItem* hashTable, int szNhood, unsigned int* neighbors,
        int* n2c, int* renumber, unsigned int bucketSize, unsigned int laneId,
        int *offsetInGlobalMem, unsigned int* newLinksOfNewComm, float* newWeightsOfNewComm, unsigned int WARP_SIZE) {


    HashItem item;
    unsigned int nrAttempt = 0;

    unsigned int wid = threadIdx.x / WARP_SIZE;
    wid = blockIdx.x * (blockDim.x / WARP_SIZE) + wid;
    HashItem dataItem;
    for (unsigned int i = laneId; i < szNhood; i = i + WARP_SIZE) {

        int neighId = neighbors[i];

        // if the neighbors[i] was negated then revert it

        int isOfNewComm = 1;
        if (neighId < 0) {
            isOfNewComm = -1; // this thread inserted it
        }

        neighbors[i] = neighId = (neighId * isOfNewComm) - 1; // (nId+1) Or -(nId+1) to nId

        item.cId = renumber[n2c[neighId]];
        /*
        if (wid == 1)
            printf("\n--------------search----------->>>> wid = %u laneId =%u neighId=%d C= %d szNhood=%d isNew= %d\n", wid, laneId, neighId, item.cId, szNhood, isOfNewComm);
         */
        int posInTable = hashSearchGPU(hashTable, &nrAttempt, bucketSize, &item);
        if (posInTable >= 0 && isOfNewComm == -1) {

            dataItem = hashTable[posInTable];


            newLinksOfNewComm[*offsetInGlobalMem] = dataItem.cId - 1;
            newWeightsOfNewComm[*offsetInGlobalMem] = dataItem.gravity;
            /*
            if (DUMP && cid == 6)
                printf("\n-------------Fetched---------->>>> wid = %u laneId =%u neighId=%d C= %d  W= %f szNhood=%d\n", wid, laneId, neighId, dataItem.cId - 1, dataItem.gravity, szNhood);
             */
            *offsetInGlobalMem = *offsetInGlobalMem + 1;
        }
    }


}

#ifdef RUNONGPU

__device__
#endif
void clrTblByWarp(HashItem* table, int bucketSz, int WarpSize, int laneId) {

    for (int i = laneId; i < bucketSz; i = i + WarpSize) {

        table[i].cId = 0;
        table[i].gravity = 0;
    }

}
/**
 * Process each member of new community to construct the neighborhood of new community 
 * @param table
 * @param links
 * @param weights
 * @param superNodes
 * @param commNodes
 * @param indices
 * @param bucketSize
 * @param laneId
 * @param wid
 * @param graphType
 */
#ifdef RUNONGPU

__device__
#endif
void processNodesOfNewComm(HashItem* table, unsigned int* links, float* weights,
        int* superNodes, int* commNodes, int* indices, int* n2c, int* renumber,
        unsigned int bucketSize, unsigned int laneId, unsigned int cid,
        int graphType, unsigned int* linksOfNewComm, float* weightsOfNewComm,
        unsigned int* nrNeighborsOfNewComms, unsigned int WARP_SIZE) {


    int startOfNewComm = superNodes[cid];
    int endOfNewComm = superNodes[cid + 1];

    //if (!laneId)printf("\nwid= %u %d %d\n", wid, startOfNewComm, endOfNewComm);

    int nrDiscovered = 0;

    for (int i = startOfNewComm; i < endOfNewComm; i++) {

        int node = commNodes[i];

        int startOfNeighood = indices[node];
        int endOfNeighood = indices[node + 1];

        float *weightsOfNeighood = NULL;

        if (graphType == WEIGHTED) {
            weightsOfNeighood = &weights[startOfNeighood];
        }

        int szNeighood = endOfNeighood - startOfNeighood;
        /*
        if (DUMP && cid == 6 && !laneId)
            printf("\n************node=%d szNeighood=%d (%d - %d) \n", node, szNeighood, endOfNeighood, startOfNeighood);
         */
        //clrTblByWarp(table, bucketSize, WARP_SIZE, laneId);
        //Hash all neighbors of a node

        hashNeighborsOfNode(cid, table, weightsOfNeighood, szNeighood,
                &links[startOfNeighood], n2c, renumber, bucketSize, laneId, &nrDiscovered, WARP_SIZE);

    }

    __syncthreads();

    //Inclusive Scan
    int myPosition = nrDiscovered;


    for (int i = 1; i <= WARP_SIZE / 2; i *= 2) {

        int lowerLaneValue = __shfl_up(myPosition, i, WARP_SIZE);
        if (laneId >= i)
            myPosition += lowerLaneValue;
    }

    __syncthreads();

    /*
    if (threadIdx.x == blockDim.x - 1 && ((cid >= 31221 && cid <= 31223) || (cid >= 34691 && cid <= 34693)))
        printf("\ncid: %d, myPosition= %d \n", cid, myPosition);
     */

    // write actual #neighbor of new community
    if (laneId == WARP_SIZE - 1) {
        nrNeighborsOfNewComms[cid + 1] = myPosition;
    }


    //if(wid==WID && laneId==WARP_SIZE-1)
    //printf("\n#Hashed=%d\n",myPosition);


    //convert to exclusive scan
    myPosition = myPosition - nrDiscovered;


    //if (wid == 1)printf("\nwid = %u, laneId = %u pos=%d\n", wid, laneId, myPosition);

    for (int i = startOfNewComm; i < endOfNewComm; i++) {

        int node = commNodes[i];
        int startOfNeighood = indices[node];

        int endOfNeighood = indices[node + 1];

        int szNeighood = endOfNeighood - startOfNeighood;
        /*
        if (wid == 1 && !laneId)
            printf("\nGoing to collect from Hash, node=%d \n", node);
         */
        collectFromHashTable(cid, table, szNeighood, &links[startOfNeighood], n2c, renumber,
                bucketSize, laneId, &myPosition, linksOfNewComm, weightsOfNewComm, WARP_SIZE);
    }

}


/**
 * Exclusive scan
 * @param nrDiscovered
 * @param nrNeighborsOfNewComms
 * @param cId
 * @return 
 */
#ifdef RUNONGPU

__device__
#endif
int blockPrefix(int nrDiscovered, unsigned int* nrNeighborsOfNewComms, int cId, unsigned int WARP_SIZE) {

    int myPosition = nrDiscovered;
    int laneId = threadIdx.x & (WARP_SIZE - 1);
    int nrWarpInBlock = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    int toReturn = -1;

    // Works for nrWarpInBlock <= WARP_SIZE
    //nrWarpInBlock > WARP_SIZE  ???  => Problem 

    if (nrWarpInBlock <= PHY_WRP_SZ) {

        //Inclusive Scan ( intra-warp within warp) gives total inserted by a warp
        for (int i = 1; i <= WARP_SIZE / 2; i *= 2) {

            int lowerLaneValue = __shfl_up(myPosition, i, WARP_SIZE);
            if (laneId >= i)
                myPosition += lowerLaneValue;
        }


        __shared__ int foundByWarps[ PHY_WRP_SZ ];
        volatile int* vmem = foundByWarps;

        //One thread of each warp writes the #found by the warp in sharedMemory
        // Last thread of each warp has this number

        if (laneId == WARP_SIZE - 1) {
            vmem[ threadIdx.x / WARP_SIZE ] = myPosition; // total found by this warp
        }

        // waits for all the warps to get done
        __syncthreads();

        /*
        if (threadIdx.x == WARP_SIZE - 1 || threadIdx.x == blockDim.x - 1) {
            if (cId == 34692)
                for (int m = 0; m < nrWarpInBlock; m++)
                    printf("\n %d %d  threadIdx.x=%d\n", m, vmem[m], threadIdx.x);
        }
        __syncthreads();
         */

        //Prefix sum on foundByWarps array
        //Each warp does the same thing 


        int toAdd = 0;
        if (laneId < nrWarpInBlock) {
            toAdd = vmem[laneId];
        }

        for (int i = 1; i <= WARP_SIZE / 2; i *= 2) {

            int lowerLaneValue = __shfl_up(toAdd, i, WARP_SIZE);

            if (laneId >= i)
                toAdd += lowerLaneValue;
        }

        //Only last thread of each warp has the #discovered by whole block
        if (threadIdx.x == blockDim.x - 1) {
            nrNeighborsOfNewComms[cId + 1] = toAdd;
        }

        /*
        // First and Last warp will print content of vmem
        if (threadIdx.x == WARP_SIZE - 1 || threadIdx.x == blockDim.x - 1) {

            if (cId == CID) {
                for (int m = 0; m < nrWarpInBlock; m++)
                    printf("\n %d %d  (threadIdx.x=%d)\n", m, vmem[m], threadIdx.x);
            }
        }

        // Each warp will print toAdd
        if ((laneId == (WARP_SIZE - 1)) && cId == CID) {
            printf("\n---> cId=%d toAdd= %d theadIdx.x= %d \n", cId, toAdd, threadIdx.x);
        }
         */
        /*
        if (threadIdx.x == blockDim.x - 1 && cId == CID)
            printf("\ncId: %d, myPosition= %d toAdd =%d \n", cId, myPosition, toAdd);
         */


        //Each thread determines global position to write its parts
        unsigned int wId = threadIdx.x / WARP_SIZE;

        if (wId >= 1) {
            myPosition = myPosition + __shfl(toAdd, (wId - 1), WARP_SIZE);
        }

        myPosition = myPosition - nrDiscovered; //Exclusive scan

        toReturn = myPosition;

    } else {
        if (threadIdx.x == 0 && blockIdx.x == 0)
            printf("\nblockPrefix Failed!! nrWarpInBlock= %d, WSz= %u\n", nrWarpInBlock, WARP_SIZE);
    }
    return toReturn;
}
#ifdef RUNONGPU

__device__
#endif
void checkTable(int cId, int bucketSize, HashItem* table) {

    int occupied = 0;

    int soughtCId = 0;
    if (cId == soughtCId) {

        __syncthreads();

        for (int m = 0; m < bucketSize; m++) {
            if (table[m].cId != FLAG_FREE) {
                occupied++;
            }
        }

        if (threadIdx.x == 0 || threadIdx.x == blockDim.x - 1)
            printf("\n #occupied %d out of %d cid= %d (tid= %d) \n", occupied, bucketSize, cId, threadIdx.x);

        __syncthreads();
    }
}
#ifdef RUNONGPU

__device__
#endif
void hashNeighbors(HashItem* hashTable, float* weightsOfNhood,
        int szNhood, unsigned int* neighbors, int* n2c, int* renumber,
        unsigned int bucketSize, unsigned int workerId, int *nrDiscoveredByMe,
        int stride, int cId, unsigned int wrpSz, int node) {



    int perThCnt = 0;
    HashItem item;
    /*
    if (cId == CID && threadIdx.x == 0)
        printf("\n cId= %d, bucketSize=%d  szNhood = %d \n", cId, bucketSize, szNhood);
     */
    //checkTable(cId, bucketSize, hashTable);
    for (unsigned int i = workerId; i < szNhood; i = i + stride) {

        perThCnt++;

        //if (blockIdx.x == 0 && threadIdx.x == 0)
        //printf("\nzszNhood = %d #stride= %d\n", szNhood, stride);
        //printf("\ni=%d neighbor[%d] =%d szNhood = %d\n", i, i, neighbors[i], szNhood);

        item.cId = renumber[n2c[neighbors[i]]];
        item.gravity = (weightsOfNhood == NULL) ? 1.0 : weightsOfNhood[i];

        //hash the item into hasTable
        /*
        if (cId == CID)
            printf("\nGoing to Insert %d with node=%f ( node= %d) \n", item.cId, item.gravity, node);
         */
        int retValue = hashInsertSimple2(hashTable, bucketSize, &item, cId, node);


        //if(blockIdx.x == 0)
        //printf("\nworkerId=%u  Hasing %d nId= %u retValue=%d bucketSize =%d \n", workerId, item.cId, neighbors[i], retValue, bucketSize);


        //returned bucketSize if inserted new record else position of record
        if (retValue == bucketSize) {
            *nrDiscoveredByMe = *nrDiscoveredByMe + 1;
        } /*else {
            printf("\n --> retValue= %d, item.cId= %d, bucketSize=%d  cid = %d \n", retValue, item.cId, bucketSize, cId);
        }*/

        /*
        if (cId == 34692 && item.cId == 43804) {
            printf("\nneigh=%d, cId(neigh)=%d, bucketSize=%d\n", neighbors[i], renumber[n2c[neighbors[i]]], bucketSize);
        }
         */

        //Negate if it was inserted as new in the table
        retValue = (retValue == bucketSize) ? -1 : 1;
        neighbors[i] = (1 + neighbors[i]) * retValue; // nId is now (nId+1) OR -(nId+1)

        //if(blockIdx.x==0)
        //printf("\n-----**------workerId=%u  Hasing %d nId= %u retValue=%d \n", workerId, item.cId, neighbors[i], retValue);

    }


    //checkTable(cId, bucketSize, hashTable);
    /*
    if (cId == CID) {

        int nrHashEntries = blockReduce(perThCnt, wrpSz);

        if (threadIdx.x == 0)
            printf("\n cId= %d perThCnt= %d szNhood = %d nrHashEntries = %d stride= %d\n", cId, perThCnt, szNhood, nrHashEntries, stride);

        nrHashEntries = blockReduce(*nrDiscoveredByMe, wrpSz);

        if (threadIdx.x == 0)
            printf("\n cId= %d perThCnt= %d szNhood = %d #discovered = %d stride= %d\n", cId, perThCnt, szNhood, nrHashEntries, stride);


    }
     */
    __syncthreads();

}
#ifdef RUNONGPU

__device__
#endif
void collectFromHash(HashItem* hashTable, int szNhood, unsigned int* neighbors,
        int* n2c, int* renumber, unsigned int bucketSize, unsigned int workerId,
        int *offsetInGlobalMem, unsigned int* newLinksOfNewComm,
        float* newWeightsOfNewComm, int stride) {


    HashItem item;
    unsigned int nrAttempt = 0;

    HashItem dataItem;

    for (unsigned int i = workerId; i < szNhood; i = i + stride) {

        int neighId = neighbors[i];

        // if the neighbors[i] was negated then revert it

        int isOfNewComm = 1;
        if (neighId < 0) {
            isOfNewComm = -1; // this thread inserted it
        }

        neighbors[i] = neighId = (neighId * isOfNewComm) - 1; // (nId+1) Or -(nId+1) to nId

        item.cId = renumber[n2c[neighId]];

        //printf("\n--------------search----------->>>> wid = %u laneId =%u neighId=%d C= %d szNhood=%d isNew= %d\n", wid, laneId, neighId, item.cId, szNhood, isOfNewComm);

        int posInTable = hashSearchGPU(hashTable, &nrAttempt, bucketSize, &item);
        if (posInTable >= 0 && isOfNewComm == -1) {

            dataItem = hashTable[posInTable];

            //hashTable[posInTable].cId = FLAG_FREE;
            //hashTable[posInTable].gravity = 0.0;

            //if(threadIdx.x==0 && blockIdx.x==0)
            //printf("\n-------------Fetched---------->>>>  workerId=%d neighId=%d C= %d szNhood=%d\n",  workerId, neighId, dataItem.cId, szNhood);

            newLinksOfNewComm[*offsetInGlobalMem] = dataItem.cId - 1;
            newWeightsOfNewComm[*offsetInGlobalMem] = dataItem.gravity;
            *offsetInGlobalMem = *offsetInGlobalMem + 1;
        }
    }
}

__global__ void prescan(float *g_odata, float *g_idata, int n) {
    extern __shared__ float temp[]; // allocated on invocation  
    int thid = threadIdx.x;
    int offset = 1;

    temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory  
    temp[2 * thid + 1] = g_idata[2 * thid + 1];

    for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree  
    {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        if (thid == 0) {
            temp[n - 1] = 0;
        } // clear the last element  
    }


    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan  
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    g_odata[2 * thid] = temp[2 * thid]; // write results to device memory  
    g_odata[2 * thid + 1] = temp[2 * thid + 1];

}


#ifdef RUNONGPU

__device__
#endif
void processCommByBlock(HashItem* table, unsigned int* links, float* weights,
        int* superNodes, int* commNodes, int* indices, int* n2c, int* renumber,
        unsigned int bucketSize, int graphType, unsigned int* linksOfNewComm,
        float* weightsOfNewComm, unsigned int* nrNeighborsOfNewComms, int cId,
        unsigned int wrpSz) {

    int startOfNewComm = superNodes[cId];
    int endOfNewComm = superNodes[cId + 1];


    //if (blockIdx.x == 0 && threadIdx.x == 0 && cId >= 0)
    //printf("\ncId=%d startOfNewComm = %d, endOfNewComm = %d \n", cId, startOfNewComm, endOfNewComm);


    int nrDiscovered = 0;

    for (int i = startOfNewComm; i < endOfNewComm; i++) {

        int node = commNodes[i];

        int startOfNeighood = indices[node];
        int endOfNeighood = indices[node + 1];

        float *weightsOfNeighood = NULL;

        if (graphType == WEIGHTED) {
            weightsOfNeighood = &weights[startOfNeighood];
        }

        int szNeighood = endOfNeighood - startOfNeighood;

        /*
        if ((threadIdx.x == 0 || threadIdx.x == blockDim.x - 1) && (cId == CID))
            printf("\n***node=%d szNeighood=%d (%d - %d) \n", node, szNeighood, endOfNeighood, startOfNeighood);
         */

        //Hash all neighbors of a node
        hashNeighbors(table, weightsOfNeighood, szNeighood,
                &links[startOfNeighood], n2c, renumber, bucketSize, threadIdx.x,
                &nrDiscovered, blockDim.x, cId, wrpSz, node);

        // Just for debugging in case each NODE is a community
        /*
        __syncthreads();
        int nrUnique = blockReduce(nrDiscovered);
        __syncthreads();

        if ((nrUnique != szNeighood) && (threadIdx.x == (blockDim.x - 1) || threadIdx.x == 0)) {
            printf("\n\nERROR: cId=%d,nrInUnique= %d, szNeighood= %d tid= %d\n\n", cId, nrUnique, szNeighood, threadIdx.x);
        }
         */
    }

    //Wait for everything to be hashed
    __syncthreads();

    int myPosition = blockPrefix(nrDiscovered, nrNeighborsOfNewComms, cId, wrpSz);

    /*
    int nrH = blockReduce(nrDiscovered);
    __syncthreads();
    
    if (threadIdx.x == 0)
        nrNeighborsOfNewComms[cId + 1] = nrH;
    
    __syncthreads();

    if (threadIdx.x == 0 && (cId == CID))
        printf("\n***cid=%d, #discovered= %d threadIdx.x = %d  \n", cId, nrH, threadIdx.x);
     */

    for (int i = startOfNewComm; i < endOfNewComm; i++) {

        int node = commNodes[i];
        int startOfNeighood = indices[node];

        int endOfNeighood = indices[node + 1];

        int szNeighood = endOfNeighood - startOfNeighood;

        /*
        if (blockIdx.x == 0 && threadIdx.x == 0)
            printf("\nGoing to collect from Hash, node=%d \n", node);
         */
        collectFromHash(table, szNeighood, &links[startOfNeighood], n2c,
                renumber, bucketSize, threadIdx.x, &myPosition, linksOfNewComm,
                weightsOfNewComm, blockDim.x);
    }

    // Make sure all threads have collected from hashTable
    __syncthreads();

    // if (threadIdx.x == 0 && blockIdx.x == 0 && cId == CID)
    //printf("\ncId=%d  |Neighborhood| =%d\n", cId, nrNeighborsOfNewComms[cId + 1]);
}

/**
 * Make sure that both blocks and grids  are ONE-DIMENSIONAL
 */
__global__
void findNewNeighodByBlock(int* super_node_ptrs, float* newWeights,
        unsigned int* newLinks, unsigned int* nrNeighborsOfNewComms,
        int* indices, float* weights, unsigned int* links, int* comms_nodes,
        int new_nb_comm, int* n2c, int* renumber, int* start_locations,
        int graphType, unsigned int bucketSize, int* candidateComms,
        int nrCandidateComms, HashItem* gblTable, int* glbTblPtrs,
        int* primes, int nrPrime, unsigned int wrpSz) {


    HashItem* blockTable = NULL;

    __shared__ HashItem blkTblShared[SHARED_TABLE_SIZE];

    //NOTE: Make sure host module has allocated at least 2 times memory than upper limit

    //int tblStart = 2 * glbTblPtrs [blockIdx.x];
    int tblStart = glbTblPtrs [blockIdx.x];
    int tblEnd = glbTblPtrs [blockIdx.x + 1];

    blockTable = &gblTable[tblStart ];

    unsigned int commIndex = blockIdx.x;

    while (commIndex < nrCandidateComms) {

        int cId = candidateComms[commIndex];
        /*
        if (cId == CID && threadIdx.x == 0)
            printf("\n$$ bId=%d cId=%d |nrCandidateComms|=%d \n", blockIdx.x, cId, nrCandidateComms);
         */
        // start location to write new neighborhood
        int gblStart = start_locations[cId];
        int gblEnd = start_locations[cId + 1];

        //USE maxSizeOfNeighborhood to decide type of hashTable;shared or global
        int maxSizeOfNeighborhood = gblEnd - gblStart;

        if (maxSizeOfNeighborhood < (SHARED_TABLE_SIZE / 2)) {

            blockTable = blkTblShared;
            bucketSize = SHARED_TABLE_SIZE;


            int nearestPrime = findPrimebyWarp(primes, nrPrime, (maxSizeOfNeighborhood * 3) / 2, wrpSz);
            /*	
            if (nearestPrime < 0 || nearestPrime > SHARED_TABLE_SIZE) // ????????????
                nearestPrime = findPrimebyWarp(primes, nrPrime, maxSizeOfNeighborhood, wrpSz);
             */

            if (nearestPrime < 0 || nearestPrime > SHARED_TABLE_SIZE) {

                if (threadIdx.x == 0 || threadIdx.x == blockDim.x - 1)
                    printf("\n---------Nearest Prime Can't be negative- or > STS--------maxSzNhd=%d \n", maxSizeOfNeighborhood);

                return;
            }

            bucketSize = nearestPrime;

        } else {


            blockTable = &gblTable[tblStart];

            int nearestPrime = findPrimebyWarp(primes, nrPrime, (maxSizeOfNeighborhood * 2) / 2, wrpSz);

            if (nearestPrime < 0)
                nearestPrime = findPrimebyWarp(primes, nrPrime, maxSizeOfNeighborhood, wrpSz);

            if (nearestPrime < 0 && (threadIdx.x == 0 || threadIdx.x == blockDim.x - 1))
                printf("\n-----------Nearest Prime Can't be negative-----------maxSzNhd=%d \n", maxSizeOfNeighborhood);

            bucketSize = nearestPrime;
        }

        if (tblStart + bucketSize > tblEnd && blockTable == &gblTable[tblStart]) {

            if (threadIdx.x == 0)
                printf("\nPROBLEM:: Gbl Tbl overflow; bktSz= %d shTblSz= %d tblStart= %d,tblEnd=%d maxSzNd= %d blkId= %u\n", bucketSize, SHARED_TABLE_SIZE, tblStart, tblEnd, maxSizeOfNeighborhood, blockIdx.x);

            return;
        }

        initByBlock(blockTable, bucketSize, threadIdx.x, blockDim.x);



        processCommByBlock(blockTable, links, weights, super_node_ptrs,
                comms_nodes, indices, n2c, renumber, bucketSize, graphType,
                &newLinks[gblStart], &newWeights[gblStart],
                nrNeighborsOfNewComms, cId, wrpSz);

        // Put Marked on unused memory


        unsigned int l = (gblStart + nrNeighborsOfNewComms[cId + 1]) + threadIdx.x;

        //if (threadIdx.x == 0 &&  nrNeighborsOfNewComms[cId + 1] > 0)
        //printf(" GMC= %d cId= %d", nrNeighborsOfNewComms[cId + 1], cId);


        for (; l < gblEnd; l = l + blockDim.x) {
            //Markers
            newLinks[l] = 2 * ((blockDim.x * gridDim.x) / wrpSz) * new_nb_comm;
            newWeights[l] = -55.55; // Marker
        }
        commIndex += gridDim.x;

        //if (blockIdx.x == 0 && threadIdx.x == 0)
        //printf("\nbId=%d commIndex=%d\n", blockIdx.x, commIndex);
        //break;
    }
}

__global__
void determineNewNeighborhood(int* super_node_ptrs, float* newWeights,
        unsigned int* newLinks, unsigned int* nrNeighborsOfNewComms, int* indices,
        float* weights, unsigned int* links, int* comms_nodes, int new_nb_comm,
        int* n2c, int* renumber, int* start_locations, int graphType,
        unsigned int bktSzLimit, int* candidateComms, int nrCandidateComms,
        unsigned int WARP_SIZE) {

    unsigned int wid = threadIdx.x / WARP_SIZE;
    unsigned int laneId = threadIdx.x & (WARP_SIZE - 1); // id in the warp


    extern __shared__ HashItem __sMem[];

    // NOTE: bucketSize == size of Table in "Warp Memory"
    HashItem* shashTable = __sMem + wid * bktSzLimit;

    initWarpHashTable(shashTable, bktSzLimit, laneId, WARP_SIZE);

    // Global warp ID and pointer in global Memory
    wid = blockIdx.x * (blockDim.x / WARP_SIZE) + wid;
    /*
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("\n#WARP = %d laneId = %u wid = %u\n", ((blockDim.x * gridDim.x) / WARP_SIZE), laneId, wid);
     */
    while (wid < nrCandidateComms) { // this warp got something to do

        //start_locations contains cumulative sum on upper bound of size of
        //new neighborhood of each new community and hence indicates the global
        //position to place neighbor of new community

        int cId = candidateComms[wid];
        int gblStart = start_locations[cId];
        int gblEnd = start_locations[cId + 1];


        int maxSizeOfNeighborhood = gblEnd - gblStart;

        //if (cId == 63 && !laneId)printf("\nwid =%u Start=%d End=%d maxSizeOfNeighborhood = %d \n", wid, gblStart, gblEnd, maxSizeOfNeighborhood);

        if (maxSizeOfNeighborhood <= bktSzLimit) {

            initWarpHashTable(shashTable, bktSzLimit, laneId, WARP_SIZE);

            processNodesOfNewComm(shashTable, links, weights, super_node_ptrs,
                    comms_nodes, indices, n2c, renumber, bktSzLimit, laneId,
                    cId, graphType, &newLinks[gblStart], &newWeights[gblStart],
                    nrNeighborsOfNewComms, WARP_SIZE);

        } else {
            if (laneId == 0)
                printf("\nmaxSizeOfNeighborhood > bucketSize!\n");
        }

        //if (maxSizeOfNeighborhood < (bucketSize * LOAD_FACTOR) / 2) {

        // Put Marked on unused memory
        unsigned int l = (gblStart + nrNeighborsOfNewComms[cId + 1]) + laneId;


        for (; l < gblEnd; l = l + WARP_SIZE) {
            // Markers
            newLinks[l] = 2 * ((blockDim.x * gridDim.x) / WARP_SIZE) * new_nb_comm;
            newWeights[l] = -55.55;
        }


        //}
        wid = wid + (blockDim.x * gridDim.x) / WARP_SIZE;
    }

}


#ifdef RUNONGPU

__device__
#endif

void copy_from_global_to_shared(int laneId, int segment_len, volatile int* dest, int* src, unsigned int WARP_SIZE) {

    for (int i = laneId; i < segment_len; i = i + WARP_SIZE) {
        dest[i] = src[i];
    }

}

__global__
void initialize_in_tot(int community_size, int* indices, unsigned int* links,
        float* weights, float* tot, float *in, int* n2c, int type, int* locks,
        unsigned int WARP_SIZE, float* wDegs) {

    unsigned int wid = threadIdx.x / WARP_SIZE;
    unsigned int laneId = threadIdx.x % WARP_SIZE; // id in the warp

    extern __shared__ int blockMemory[];

    int* myMemory = blockMemory + (CHUNK_PER_WARP + 1 + CHUNK_PER_WARP) * wid;
    //(CHUNK_PER_WARP+1) indices
    //CHUNK_PER_WARP  n2c(nodes)
    volatile int* warpMemory = myMemory;
    volatile int* warp_n2c = myMemory + (CHUNK_PER_WARP + 1);

    // local warp id

    if (!laneId && PRINTALL) {
        printf(" @tid: %d \n", (wid * WARP_SIZE + laneId));
    }

    // Global warp ID
    wid = blockIdx.x * (blockDim.x / WARP_SIZE) + wid;

    int num_vertex_to_process = community_size - wid*CHUNK_PER_WARP;


    if ((wid + 1) * CHUNK_PER_WARP <= community_size) {
        num_vertex_to_process = CHUNK_PER_WARP;
    }



    if (num_vertex_to_process > 0) {
        // copy indices from global memory to shared memory

        if ((wid * CHUNK_PER_WARP + num_vertex_to_process) > community_size)
            if (!laneId) printf("\n 1#**can't happen**\n");

        copy_from_global_to_shared(laneId, num_vertex_to_process + 1, warpMemory, &indices[wid * CHUNK_PER_WARP], WARP_SIZE);
        copy_from_global_to_shared(laneId, num_vertex_to_process, warp_n2c, &n2c[wid * CHUNK_PER_WARP], WARP_SIZE);

    }

    /*
    if (!laneId)
        printf("\n %d : %d \n", wid, num_vertex_to_process);
     */

    for (int vid_index_in_warp = 0; vid_index_in_warp < num_vertex_to_process; vid_index_in_warp++) {


        int node = wid * CHUNK_PER_WARP + vid_index_in_warp;


        unsigned int start_of_neighbors = warpMemory[vid_index_in_warp];

        unsigned int end_of_neighbors = warpMemory[vid_index_in_warp + 1];

        int comm_of_node = warp_n2c[vid_index_in_warp];

        if (node != comm_of_node) {
            if (!laneId && !wid) printf("\nCrack on a crakcer !\n");
        }

        int nr_neighbor = (end_of_neighbors - start_of_neighbors);


        float *weightsMem = NULL;

        if (type == WEIGHTED) {

            weightsMem = &weights[start_of_neighbors];
        }

        float weighted_degree_of_node = wDegs[node];
        /*
        float weighted_degree_of_node = comWDegOfNode(laneId, nr_neighbor, weightsMem, type, WARP_SIZE);
        if( wDegs[node] !=  weighted_degree_of_node){
                if(!laneId)printf("\nWDegs PROBLEM; init in tot\n");
        }
         */
        if (!laneId) {
            atomicAdd(&tot[ comm_of_node], weighted_degree_of_node);

        }

    }
}

template <typename Tdata >
#ifdef RUNONGPU

__device__
#endif
Tdata findMaxPerWarp(unsigned int laneId, unsigned int nrElements,
        Tdata * inputData, unsigned int WARP_SIZE) {

    Tdata maxElement = (Tdata) (1 << (7 * sizeof (Tdata)));
    maxElement = -maxElement;

    for (unsigned int i = laneId; i < nrElements; i = i + WARP_SIZE) {
        if (inputData[i] > maxElement)
            maxElement = inputData[i];
    }
    for (unsigned int i = WARP_SIZE / 2; i >= 1; i = i / 2) {
        Tdata received = __shfl_xor(maxElement, i, WARP_SIZE);
        if (received > maxElement)
            maxElement = received;
    }
    //printf("\nlaneId= %u, max= %d \n", laneId, (int) maxElement);
    return maxElement;

}

__global__
void computeMaxDegreeForWarps(int * indices, int *maxDegreePerWarp,
        int *nrUniDegPerWarp, unsigned int communitySize, unsigned int WARP_SIZE) {

    unsigned int wid = threadIdx.x / WARP_SIZE;
    unsigned int laneId = threadIdx.x % WARP_SIZE; // id in the warp

    extern __shared__ int blockMemory[];

    int* myMemory = blockMemory + (CHUNK_PER_WARP) * wid;

    volatile int* warpMemory = myMemory; // Memory dedicated to current warp


    // Global warp ID
    wid = blockIdx.x * (blockDim.x / WARP_SIZE) + wid;

    int numVertexToProcess = communitySize - wid*CHUNK_PER_WARP;

    if ((wid + 1) * CHUNK_PER_WARP <= communitySize) {
        numVertexToProcess = CHUNK_PER_WARP;
    }

    if (numVertexToProcess > 0) {

        /*
        // copy indices from global memory to shared memory
        copy_from_global_to_shared(laneId, numVertexToProcess + 1, warpMemory,
                &indices[wid * CHUNK_PER_WARP]);
         */

        /*
        if (!laneId)
            printf("\nwarpID= %u,  (index in sharedMem) upto= %d  numVertexToProcess=%d \n",
                    wid, (CHUNK_PER_WARP * wid + CHUNK_PER_WARP), numVertexToProcess);
         */
        //Initialize memory to zero
        for (int l = laneId; l < CHUNK_PER_WARP; l = l + WARP_SIZE) {
            warpMemory[l] = 0;
            //if (laneId <= WARP_SIZE - 1)printf("\nwarpId=%u laneId= %u,  l=%d  warpMem[%d]=%d \n", wid, laneId, l, l, warpMemory[l]);
        }

        int uniDegCounter = 0;
        // compute degree of vertices from indices array
        for (int l = laneId; l < numVertexToProcess; l = l + WARP_SIZE) {
            int startOfNeigh = indices[wid * CHUNK_PER_WARP + l];
            int endOfNeigh = indices[wid * CHUNK_PER_WARP + l + 1]; // It's NOT a problem !!
            if (warpMemory[l] < (endOfNeigh - startOfNeigh + 1)) {
                warpMemory[l] = (endOfNeigh - startOfNeigh + 1);
            }

            if ((endOfNeigh - startOfNeigh) == 1)
                uniDegCounter = uniDegCounter + 1;
        }

        for (int i = WARP_SIZE / 2; i >= 1; i = i / 2) {

            uniDegCounter += __shfl_xor(uniDegCounter, i, WARP_SIZE);
        }

        if (!laneId)nrUniDegPerWarp[wid] = uniDegCounter;
        // determine max element by reading shared memory
        maxDegreePerWarp[wid] = findMaxPerWarp(laneId, numVertexToProcess, warpMemory, WARP_SIZE);
    }

}

