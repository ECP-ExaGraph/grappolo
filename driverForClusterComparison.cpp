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
#include "utilityClusteringFunctions.h"

using namespace std;

int main(int argc, char** argv) {

    if (argc < 3) {
        printf("================================================================================================\n");
        printf("Usage: %s <GroundTruth Communities> <Output Communities> \n", argv[0]);
        printf("================================================================================================\n");
        exit(-1);
    }
    ifstream File;
    //Parse file with ground truth communities:
    long N1=0;
    vector<long> truthCommunity;
    File.open(argv[1]);
    long temp;
    // TODO FIXME provide choices for 1-based 
    // and whether to read 1/2 tokens per line 
    while(!File.eof()) {
        File >> temp;
        truthCommunity.push_back((long)temp);
    }
    cout << "\n";
    File.close();
    N1 = truthCommunity.size();
    N1--;
    cout<< "Parsed ground truth file with " << N1 << " elements\n";
    
    //Parse file with output community:
    long N2=0;
    vector<long> outputCommunity;
    File.open(argv[2]);
    while(!File.eof()) {
        File >> temp;
        outputCommunity.push_back((long)temp);
    }
     cout << "\n";
    File.close();
    N2 = outputCommunity.size();
    N2--;
    cout<< "Parsed output file with " << N2 << " elements\n";
    
    //Call the cluster comparison function:
    computeCommunityComparisons(truthCommunity, N1, outputCommunity, N2);
   
  return 0;
}
