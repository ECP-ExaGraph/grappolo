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

#include <fstream>
#include <algorithm>
#include <iostream>
#include "communityGPU.h"

void Community::set_new_graph_as_current() {

    g.indices.clear();
    g.links.clear();
    g.weights.clear();

    g.indices.resize(g_next.nb_nodes + 1);
    g.links.resize(g_next.nb_links);
    g.weights.resize(g_next.nb_links);

    /*
    {
        std::cout << std::endl << "[Before Exchange] g.size(): " << g.nb_nodes << std::endl;
    }
     */
    g = g_next;

    //deep copy or sallow copy ?
    {

        g_next.indices.clear();
        g_next.links.clear();
        g_next.weights.clear();
    }
    /*
    {
        std::cout << std::endl << "[After Exchange] g.size(): " << g.nb_nodes << " #E: " << g.links.size() << std::endl;
    }
     */

    //std::cout << "...............Set New Graph as Current.................................." << std::endl;

    int sc;
    sc = 0; //std::cin>>sc;
    if (sc > 0) {
        print_vector(g.indices, "g.indices( after exchanging): ");
        print_vector(g.weights, "g.weights( after exchanging): ");
        print_vector(g.links, "g.links( after exchanging): ");
    }

    community_size = g.nb_nodes;

}

void Community::readPrimes(std::string filename) {

    std::ifstream filePtr(filename.c_str());

    if (filePtr.is_open()) {

        int nrPrimes, aPrimeNum;

        filePtr>>nrPrimes;

        assert(nrPrimes > 0);

        std::cout << "Reading " << nrPrimes << " prime numbers." << std::endl;

        //Read primes in host memory
        hostPrimes = new int [nrPrimes];
        int index = 0;
        while (filePtr >> aPrimeNum) {

            hostPrimes[index++] = aPrimeNum;
            if (index >= nrPrimes)
                break;
            //std::cout << aPrimeNum << " ";
        }

        std::cout << std::endl;

        assert(nrPrimes == index);
        nb_prime = nrPrimes;

        //Copy prime numbers to device memory
        devPrimes.resize(nb_prime);
        thrust::copy(hostPrimes, hostPrimes + nb_prime, devPrimes.begin());

    } else {
        std::cout << "Can't open file containing prime numbers." << std::endl;
    }
    return;
}

