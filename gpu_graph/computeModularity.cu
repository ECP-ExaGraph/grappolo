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

struct my_modularity_functor_2 {
    double m2;

#ifdef RUNONGPU

    __host__ __device__
#endif

    my_modularity_functor_2(double _m2) : m2(_m2) {
    }

#ifdef RUNONGPU

    __host__ __device__
#endif

    double operator()(const float& x, const float& y) {
        return ((double) x / m2 - ((double) y / m2)*((double) y / m2));
    }

};

double Community::modularity(thrust::device_vector<float>& tot, thrust::device_vector<float>& in) { // put i=j in equation (1)

    float q = 0.;
    float m2 = (float) g.total_weight;

    std::cout << "m2: " << m2 << std::endl;

    /*
    thrust::host_vector<float> in_ = in;
    thrust::host_vector<float> tot_ = tot;
    //std::cout << "QQ:" << std::endl;
    for (int i = 0; i < community_size; i++) {
        if (tot_[i] > 0) {
            //std::cout << in_[i] << " " << tot_[i] << " " << m2 << " : ";
            q += in_[i] / m2 - (tot_[i] / m2)*(tot_[i] / m2);
        }
    }
    //std::cout << std::endl;
    return q;
     */

    /*
    double q = 0.;
    double m2 = (double) g.total_weight;

    for (int i = 0; i < size; i++) {
        if (tot[i] > 0)
            q += (double) in[i] / m2 - ((double) tot[i] / m2)*((double) tot[i] / m2);
    }

    return q;
     */

    bool hostPrint = false;

    if (hostPrint) {
        std::cout << std::endl << " Inside  modularity() " << std::endl;

        thrust::copy(tot.begin(), tot.end(), std::ostream_iterator<float>(std::cout, " "));
        std::cout << std::endl;

        thrust::copy(in.begin(), in.end(), std::ostream_iterator<float>(std::cout, " "));
        std::cout << std::endl;
    }

    int sc;
    //std::cout << community_size << " |in|:" << in.size() << " |tot|:" << tot.size() << std::endl;

    sc = 0; //std::cin>>sc;
    thrust::device_vector<double> result_array(community_size, 0.0);
    sc = 0; //std::cin>>sc;

    thrust::transform(thrust::device, in.begin(), in.end(), tot.begin(), result_array.begin(), my_modularity_functor_2(m2));

    q = thrust::reduce(thrust::device, result_array.begin(), result_array.end(), (double) 0, thrust::plus<double>());
    result_array.clear();
    return q;
}

