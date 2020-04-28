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

#ifndef COMMUNITYGPU_H
#define	COMMUNITYGPU_H

#include "graphGPU.h"
#include "cuda.h"
#include"cuda_runtime_api.h"
#include "graphHOST.h"

#include"commonconstants.h"
#include"hostconstants.h"

#include"myutility.h"
#include"thrust/transform_reduce.h"
#include"thrust/functional.h"
#include"thrust/execution_policy.h"
#include"thrust/device_vector.h"
#include"thrust/reduce.h"
#include"thrust/sequence.h"
#include"thrust/transform.h"
#include"thrust/iterator/transform_iterator.h"
#include "thrust/iterator/counting_iterator.h"

#include"thrust/extrema.h"
#include"thrust/fill.h"
#include"string"

struct Community {
    int community_size;

    thrust::device_vector<int> n2c;

    thrust::device_vector<int> n2c_new;


    thrust::device_vector<int> comm_nodes;
    thrust::device_vector<int> pos_ptr_of_new_comm;

    int* hostPrimes;
    thrust::device_vector<int> devPrimes;
    int nb_prime;

    int number_pass;

    double min_modularity;

    GraphGPU g;
    GraphGPU g_next;

    //
    Community(const GraphHOST& input_graph, int nb_pass, double min_mod);

    double modularity(thrust::device_vector<float> &tot, thrust::device_vector<float> &in);
    double one_level(double init_mod, bool isLastRound);
    double one_levelGaussSeidel(double init_mod, bool isLastRound, int minSize,
            double easyThreshold, bool isGauss, cudaStream_t *streams,
            int nrStreams, cudaEvent_t &start, cudaEvent_t &stop);

    void remove(int node, int comm, double dnodecomm);


    void compute_next_graph(cudaStream_t *streams, int nrStreams,
            cudaEvent_t &start, cudaEvent_t &stop);

    void set_new_graph_as_current();
    void gatherStatistics(bool isPreprocessingStep = false);
    void readPrimes(std::string filename);
    void preProcess();

};

struct my_modularity_functor {
    double m2;

#ifdef RUNONGPU

    __host__ __device__
#endif

    my_modularity_functor(double _m2) : m2(_m2) {
    }

#ifdef RUNONGPU

    __host__ __device__
#endif

    double operator()(const float& x, const float& y) {
        return (y > 1.0)* ((double) x / m2 - ((double) y / m2)*((double) y / m2));
    }

};

template<typename In_type, typename Out_type>
struct IsGreaterThanLimit : public thrust::unary_function<In_type, Out_type> {
    In_type limit;

#ifdef RUNONGPU

    __host__ __device__
#endif
    IsGreaterThanLimit(In_type _limit) : limit(_limit) {
    }

#ifdef RUNONGPU

    __host__ __device__
#endif
    Out_type operator()(In_type x) {
        return (Out_type) (x > limit);
    }
};

template<typename In_type, typename Out_type>
struct IsLessLimit : public thrust::unary_function<In_type, Out_type> {
    In_type limit;

#ifdef RUNONGPU

    __host__ __device__
#endif
    IsLessLimit(In_type _limit) : limit(_limit) {
    }

#ifdef RUNONGPU

    __host__ __device__
#endif
    Out_type operator()(In_type x) {
        return (Out_type) (x < limit);
    }
};

template<typename In_type, typename Out_type>
struct NonZeroAndLessLimit : public thrust::unary_function<In_type, Out_type> {
    In_type limit;

#ifdef RUNONGPU

    __host__ __device__
#endif
    NonZeroAndLessLimit(In_type _limit) : limit(_limit) {
    }

#ifdef RUNONGPU

    __host__ __device__
#endif
    Out_type operator()(In_type x) {
        return (Out_type) (x < limit && x > 0);
    }
};

template<typename In_type, typename Out_type>
struct IsInRange : public thrust::unary_function<In_type, Out_type> {
    In_type lowerLimit;
    In_type upperLimit;

#ifdef RUNONGPU

    __host__ __device__
#endif
    IsInRange(In_type _llimit, In_type _ulimit) : lowerLimit(_llimit), upperLimit(_ulimit) {
    }

#ifdef RUNONGPU

    __host__ __device__
#endif
    Out_type operator()(In_type x) {
        return (Out_type) (lowerLimit <= x && x <= upperLimit);
    }
};

template<typename T>
struct IsGreaterThanZero : public thrust::unary_function<T, T> {
    T myzero;

#ifdef RUNONGPU

    __host__ __device__
#endif

    IsGreaterThanZero(T _myzero) : myzero(_myzero) {
    }

#ifdef RUNONGPU

    __host__ __device__
#endif
    T operator()(T x) {

        return (T) (x > myzero);

    }
};

template<typename T>
struct subtract_constant_functor : public thrust::unary_function< T, T> {
    T const_to_subtract;

#ifdef RUNONGPU

    __host__ __device__
#endif

    subtract_constant_functor(T _to_sub) : const_to_subtract(_to_sub) {
    }

#ifdef RUNONGPU

    __host__ __device__
#endif

    T operator()(T x) {
        return (x - const_to_subtract);
    }

};

template<typename T_from, typename T_to>
struct cast_functor : public thrust::unary_function<T_from, T_to> {
    //__host__ __device__

#ifdef RUNONGPU

    __host__ __device__
#endif

    T_to operator()(T_from from) {
        return (T_to) from;
    }
};

// community id , (a>0)*b-1

template<typename T>
struct Community_ID_By_Prefix_Sum : public thrust::binary_function< T, T, T> {
#ifdef RUNONGPU

    __host__ __device__
#endif
    T operator()(T a, T b) {

        return (a > (T) 0)*b - 1;

    }
};

template<typename In_type, typename Out_type>
struct Is_Non_Negative {
#ifdef RUNONGPU

    __host__ __device__
#endif

    bool operator()(const In_type x) {
        return !(x < (In_type) 0);
    }
};

template<class T1, class T2>
void print_vector(const thrust::device_vector<T1, T2>& dateVector, std::string title, std::string prefix = "") {
    //return;
    std::cout << std::endl << title << std::endl;
    std::cout << prefix << std::endl;
    thrust::copy(dateVector.begin(), dateVector.end(), std::ostream_iterator<T1>(std::cout, " "));
    std::cout << std::endl;

}
void report_time(cudaEvent_t start, cudaEvent_t stop, std::string moduleName);

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}


#endif	/* COMMUNITYGPU_H */
