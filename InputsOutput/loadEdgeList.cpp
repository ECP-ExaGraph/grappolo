#include "input_output.h"
#include "defs.h"
#include "sstream"
#include "utilityStringTokenizer.hpp"

void parse_DirectedEdgeList(dGraph * G, char *fileName) {
    printf("Parsing Directed Edge List formatted file as a directed graph...\n");
    printf("Assumes no information is provided and that vertices are numbered contiguous ...\n");
    int nthreads = 0;
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    printf("parse_DirectedEdgeList: Number of threads: %d\n ", nthreads);
    
    double time1, time2;
    FILE *file = fopen(fileName, "r");
    if (file == NULL) {
        printf("Cannot open the input file: %s\n",fileName);
        exit(1);
    }
    long NV=0, NE=0;
    long nv1, nv2;
    //Count number of vertices and edges
    while(!feof(file))
    {
        fscanf(file, "%ld %ld", &nv1, &nv2);
        if(nv1 > NV)
            NV = nv1;
        if(nv2 > NV)
            NV = nv2;
        NE++;
    }
#if defined(DEL_ZERO_BASED)
    NV++; 
#endif
    //NE--;
    fclose(file);
    file = fopen(fileName, "r");
    printf("|V|= %ld, |E|= %ld \n", NV, NE);
    //printf("Weights will be converted to positive numbers.\n");
    /*---------------------------------------------------------------------*/
    /* Read edge list: U V W                                             */
    /*---------------------------------------------------------------------*/
    edge *tmpEdgeList = (edge *) malloc( NE * sizeof(edge)); //Every edge stored ONCE
    assert( tmpEdgeList != NULL);
    long Si, Ti;
    double Twt = 1.0;
    time1 = omp_get_wtime();
    long mycount = 0;
    for (long i = 0; i < NE; i++) {
        fscanf(file, "%ld %ld", &Si, &Ti);
        //fscanf(file, "%ld %ld %lf", &Si, &Ti, &Twt);
#if defined(DEL_ZERO_BASED)
#else
        Si--;
        Ti--;
#endif
        assert((Si >= 0)&&(Si < NV));
        assert((Ti >= 0)&&(Ti < NV));
        tmpEdgeList[mycount].head   = Si;  //The S index: Zero-based indexing
        tmpEdgeList[mycount].tail   = Ti;  //The T index: Zero-based indexing
        tmpEdgeList[mycount].weight = Twt; //Make it positive and cast to Double
        mycount++;
        //printf("%d %d\n",Si,Ti);
    }//End of outer for loop
    fclose(file); //Close the file
    time2 = omp_get_wtime();
    printf("Done reading from file: NE= %ld. Time= %lf\n", NE, time2-time1);
    
    ///////////
    time1 = omp_get_wtime();
    long *edgeListPtrOut = (long *)  malloc((NV+1) * sizeof(long)); //Outgoing
    assert(edgeListPtrOut != NULL);
    long *edgeListPtrIn  = (long *)  malloc((NV+1) * sizeof(long)); //Incoming
    assert(edgeListPtrIn != NULL);
    edge *edgeListOut = (edge *) malloc(NE * sizeof(edge)); //Outgoing
    assert( edgeListOut != NULL);
    edge *edgeListIn  = (edge *) malloc(NE * sizeof(edge)); //Incoming
    assert( edgeListIn != NULL);
    time2 = omp_get_wtime();
    printf("Time for allocating memory for storing graph = %lf\n", time2 - time1);
    
#pragma omp parallel for
    for (long i=0; i <= NV; i++)
        edgeListPtrOut[i] = 0; //For first touch purposes
#pragma omp parallel for
    for (long i=0; i <= NV; i++)
        edgeListPtrIn[i] = 0; //For first touch purposes
    
    //////Build the EdgeListPtr Array: Cumulative addition
    time1 = omp_get_wtime();
    //#pragma omp parallel for
    for(long i=0; i<NE; i++) {
        __sync_fetch_and_add(&edgeListPtrOut[tmpEdgeList[i].head+1], 1); //Leave 0th position intact
        __sync_fetch_and_add(&edgeListPtrIn[tmpEdgeList[i].tail+1], 1); //Leave 0th position intact
    }
    for (long i=0; i<NV; i++) {
        edgeListPtrOut[i+1] += edgeListPtrOut[i]; //Prefix Sum
        edgeListPtrIn[i+1]  += edgeListPtrIn[i]; //Prefix Sum:
    }
    //The last element of Cumulative will hold the total number of characters
    time2 = omp_get_wtime();
    printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
    assert(NE == edgeListPtrOut[NV]);
    assert(NE == edgeListPtrIn[NV]);
    printf("*********** (%ld)\n", NV);
    
    //time1 = omp_get_wtime();
    //Keep track of how many edges have been added for a vertex:
    long  *addedOut  = (long *)  malloc( NV  * sizeof(long));
    assert( addedOut != NULL);
    long  *addedIn  = (long *)  malloc( NV  * sizeof(long));
    assert( addedIn != NULL);
    
#pragma omp parallel for
    for (long i = 0; i < NV; i++) {
        addedOut[i] = 0;
        addedIn[i] = 0;
    }
    
    printf("About to build edgeList...\n");
    //Build the edgeList from edgeListTmp:
#pragma omp parallel for
    for(long i=0; i<NE; i++) {
        long head      = tmpEdgeList[i].head;
        long tail      = tmpEdgeList[i].tail;
        double weight  = tmpEdgeList[i].weight;
        ///Add the edges: Outgoing and incoming
        long Where = edgeListPtrOut[head] + __sync_fetch_and_add(&addedOut[head], 1);
        edgeListOut[Where].head = head;
        edgeListOut[Where].tail = tail;
        edgeListOut[Where].weight = weight;
        Where = edgeListPtrIn[head] + __sync_fetch_and_add(&addedIn[tail], 1);
        edgeListIn[Where].head = tail;
        edgeListIn[Where].tail = head;
        edgeListIn[Where].weight = weight;
    }
    //time2 = omp_get_wtime();
    printf("Time for building edgeList = %lf\n", time2 - time1);
    
    G->numVertices  = NV;
    G->numEdges     = NE;
    G->edgeListPtrsOut = edgeListPtrOut;
    G->edgeListOut     = edgeListOut;
    G->edgeListPtrsIn = edgeListPtrIn;
    G->edgeListIn     = edgeListIn;
    
    //Clean up*/
    free(tmpEdgeList);
    free(addedOut);
    free(addedIn);
    
}//End of parse_DirectedEdgeList()

void parse_UndirectedEdgeList(graph * G, char *fileName) {
    printf("Parsing a SingledEdgeList formatted file as a general graph...\n");
    printf("WARNING: Assumes that the graph is undirected -- an edge is stored once.\n");
    int nthreads = 0;
    
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    
    double time1, time2;
    FILE *file = fopen(fileName, "r");
    if (file == NULL) {
        printf("Cannot open the input file: %s\n",fileName);
        exit(1);
    }
    
    long NV=0, NE=0;
    long nv1, nv2;
    while(!feof(file))
    {
        fscanf(file, "%ld %ld", &nv1, &nv2);
        if(nv1 > NV)
            NV = nv1;
        if(nv2 > NV)
            NV = nv2;
        NE++;
    }
#if defined(DEL_ZERO_BASED)
    NV++;
#endif
    NE--;
    NE*=2;
    fclose(file);
    
    file = fopen(fileName, "r");
    printf("|V|= %ld, |E|= %ld \n", NV, NE);
    printf("Weights will be converted to positive numbers.\n");
    /*---------------------------------------------------------------------*/
    /* Read edge list: a U V W                                             */
    /*---------------------------------------------------------------------*/
    edge *tmpEdgeList = (edge *) malloc( NE * sizeof(edge)); //Every edge stored ONCE
    assert( tmpEdgeList != NULL);
    long Si, Ti;
    double Twt;
    time1 = omp_get_wtime();
    for (long i = 0; i < NE; i++) {
        fscanf(file, "%ld %ld", &Si, &Ti);
#if defined(DEL_ZERO_BASED)
#else
        Si--;
        Ti--;
#endif
        assert((Si >= 0)&&(Si < NV));
        assert((Ti >= 0)&&(Ti < NV));
        tmpEdgeList[i].head   = Si;       //The S index
        tmpEdgeList[i].tail   = Ti;    //The T index: Zero-based indexing
        tmpEdgeList[i].weight = 1; //Make it positive and cast to Double
        i++;
        tmpEdgeList[i].head = Ti;
        tmpEdgeList[i].tail = Si;
        tmpEdgeList[i].weight = 1;
    }//End of outer for loop
    printf("%d %d\n",Si,Ti);
    fclose(file); //Close the file
    time2 = omp_get_wtime();
    printf("Done reading from file: NE= %ld. Time= %lf\n", NE, time2-time1);
    
    ///////////
    time1 = omp_get_wtime();
    long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
    assert(edgeListPtr != NULL);
    edge *edgeList = (edge *) malloc( NE * sizeof(edge)); //Every edge stored twice
    assert( edgeList != NULL);
    time2 = omp_get_wtime();
    printf("Time for allocating memory for storing graph = %lf\n", time2 - time1);
    
#pragma omp parallel for
    for (long i=0; i <= NV; i++)
        edgeListPtr[i] = 0; //For first touch purposes
    
    //////Build the EdgeListPtr Array: Cumulative addition
    time1 = omp_get_wtime();
#pragma omp parallel for
    for(long i=0; i<NE; i++) {
        __sync_fetch_and_add(&edgeListPtr[tmpEdgeList[i].head+1], 1); //Leave 0th position intact
    }
    for (long i=0; i<NV; i++) {
        edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
        //printf("%d ",edgeListPtr[i]);
    }
    //The last element of Cumulative will hold the total number of characters
    time2 = omp_get_wtime();
    printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
    printf("Sanity Check: |E| = %ld, edgeListPtr[NV]= %ld\n", NE, edgeListPtr[NV]);
    printf("*********** (%ld)\n", NV);
    
    //time1 = omp_get_wtime();
    //Keep track of how many edges have been added for a vertex:
    printf("About to allocate for added vector: %ld\n", NV);
    long  *added  = (long *)  malloc( NV  * sizeof(long));
    printf("Done allocating memory fors added vector\n");
    assert( added != NULL);
#pragma omp parallel for
    for (long i = 0; i < NV; i++)
        added[i] = 0;
    
    printf("About to build edgeList...\n");
    //Build the edgeList from edgeListTmp:
#pragma omp parallel for
    for(long i=0; i<NE; i++) {
        long head      = tmpEdgeList[i].head;
        long tail      = tmpEdgeList[i].tail;
        double weight  = tmpEdgeList[i].weight;
        
        long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);
        edgeList[Where].head = head;
        edgeList[Where].tail = tail;
        edgeList[Where].weight = weight;
    }
    //time2 = omp_get_wtime();
    printf("Time for building edgeList = %lf\n", time2 - time1);
    
    G->sVertices    = NV;
    G->numVertices  = NV;
    G->numEdges     = NE/2;
    G->edgeListPtrs = edgeListPtr;
    G->edgeList     = edgeList;
    
    free(tmpEdgeList);
    free(added);
    
}//End of parse_UndirectedEdgeList()

// simple 0-based directed edge list format 
// with weight (processed as-is)
// e.g:
// 3010 10890
// 0 1486 0.006900
// 0 1492 17.289540
// 0 1504 5.763179
// ...
// 1486 0 0.006900
// 1492 0 17.289540
// 1504 0 5.763179
// ...
void parse_UndirectedEdgeListWeighted(graph * G, char *fileName) {
    printf("Parsing a SingledEdgeList formatted file as a general graph...\n");
    printf("WARNING: Assumes that the graph is undirected -- an edge is stored once.\n");
    int nthreads = 0;
    
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    
    double time1, time2;
    FILE *file = fopen(fileName, "r");
    if (file == NULL) {
        printf("Cannot open the input file: %s\n",fileName);
        exit(1);
    }

    // read number of vertices and edges
    long NV=0, NE=0;
    fscanf(file, "%ld %ld", &NV, &NE);
    
    printf("|V|= %ld, |E|= %ld \n", NV, NE);
    printf("Weights will be processed as-is.\n");
    /*---------------------------------------------------------------------*/
    /* Read edge list: U V W                                             */
    /*---------------------------------------------------------------------*/
    edge *tmpEdgeList = (edge *) malloc( NE * sizeof(edge)); //Every edge stored ONCE
    assert( tmpEdgeList != NULL);
    long Si, Ti;
    double Twt = 1.0;
    time1 = omp_get_wtime();
    for (long i = 0; i < NE; i++) {
#if defined(ALL_WEIGHTS_ONE)
        fscanf(file, "%ld %ld", &Si, &Ti);
#else
        fscanf(file, "%ld %ld %lf", &Si, &Ti, &Twt);
#endif
#if defined(PRINT_PARSED_GRAPH)
        printf("%ld %ld %lf\n", Si, Ti, Twt);
#endif
        assert((Si >= 0)&&(Si < NV));
        assert((Ti >= 0)&&(Ti < NV));
        tmpEdgeList[i].head   = Si;          //The S index
        tmpEdgeList[i].tail   = Ti;          //The T index: Zero-based indexing
#if defined(ALL_WEIGHTS_ONE)
        tmpEdgeList[i].weight = 1.0; //Make it positive and cast to Double?
#else
        tmpEdgeList[i].weight = (double)Twt; //Make it positive and cast to Double?
#endif
    }//End of outer for loop
    fclose(file); //Close the file
    time2 = omp_get_wtime();
    printf("Done reading from file: NE= %ld. Time= %lf\n", NE, time2-time1);
    
    ///////////
    time1 = omp_get_wtime();
    long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
    assert(edgeListPtr != NULL);
    edge *edgeList = (edge *) malloc( NE * sizeof(edge)); //Every edge stored once
    assert( edgeList != NULL);
    time2 = omp_get_wtime();
    printf("Time for allocating memory for storing graph = %lf\n", time2 - time1);
    
#pragma omp parallel for
    for (long i=0; i <= NV; i++)
        edgeListPtr[i] = 0; //For first touch purposes
    
    //////Build the EdgeListPtr Array: Cumulative addition
    time1 = omp_get_wtime();
#pragma omp parallel for
    for(long i=0; i<NE; i++) {
        __sync_fetch_and_add(&edgeListPtr[tmpEdgeList[i].head+1], 1); //Leave 0th position intact
    }
    for (long i=0; i<NV; i++) {
        edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
    }
    //The last element of Cumulative will hold the total number of characters
    time2 = omp_get_wtime();
    printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
    printf("Sanity Check: |E| = %ld, edgeListPtr[NV]= %ld\n", NE, edgeListPtr[NV]);
    printf("*********** (%ld)\n", NV);
    
    time1 = omp_get_wtime();
    //Keep track of how many edges have been added for a vertex:
    printf("About to allocate for added vector: %ld\n", NV);
    long  *added  = (long *)  malloc( NV  * sizeof(long));
    printf("Done allocating memory fors added vector\n");
    assert( added != NULL);
#pragma omp parallel for
    for (long i = 0; i < NV; i++)
        added[i] = 0;
    
    printf("About to build edgeList...\n");
    //Build the edgeList from edgeListTmp:
#pragma omp parallel for
    for(long i=0; i<NE; i++) {
        long head      = tmpEdgeList[i].head;
        long tail      = tmpEdgeList[i].tail;
        double weight  = tmpEdgeList[i].weight;
        
        long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);
        edgeList[Where].head = head;
        edgeList[Where].tail = tail;
        edgeList[Where].weight = weight;
    }
    time2 = omp_get_wtime();
    printf("Time for building edgeList = %lf\n", time2 - time1);
    
    G->sVertices    = NV;
    G->numVertices  = NV;
    G->numEdges     = NE;
    G->edgeListPtrs = edgeListPtr;
    G->edgeList     = edgeList;
    
    free(tmpEdgeList);
    free(added);
    
}//End of parse_DirectedEdgeListWeighted()

void parse_UndirectedEdgeListDarpaHive(graph * G, char *fileName) {
    printf("Within parse_UndirectedEdgeListDarpaHive()\n");
    printf("Parsing a SingledEdgeList formatted file as a general graph...\n");
    printf("WARNING: Assumes that the graph is directed -- an edge is stored only once.\n");
    printf("WARNING: Assumes that nodes are 1-based\n");
    int nthreads = 0;
    
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    
    double time1, time2;
    FILE *file = fopen(fileName, "r");
    if (file == NULL) {
        printf("Cannot open the input file: %s\n",fileName);
        exit(1);
    }
    
    long NV=-1, NE=0;
    long nv1, nv2, wt;
    while(!feof(file))
    {
        fscanf(file, "%ld %ld %ld", &nv1, &nv2, &wt);
        //printf("%ld %ld %ld\n", nv1, nv2, wt);
        if(nv1 > NV)
            NV = nv1;
        if(nv2 > NV)
            NV = nv2;
        NE++;
    }
    NE--;
    //NE*=2;
    fclose(file);
    
    file = fopen(fileName, "r");
    printf("|V|= %ld, |E|= %ld \n", NV, NE);
    printf("Weights will be read from the file.\n");
    /*---------------------------------------------------------------------*/
    /* Read edge list: a U V W                                             */
    /*---------------------------------------------------------------------*/
    edge *tmpEdgeList = (edge *) malloc( (2*NE+1) * sizeof(edge)); //Every edge will be stored twice; add an extra line for empty space
    assert( tmpEdgeList != NULL);
    long Si, Ti, Wi;
    time1 = omp_get_wtime();
    long i = 0;
    while(!feof(file)) {
        fscanf(file, "%ld %ld %ld", &Si, &Ti, &Wi);
        Si--; Ti--; //Numbers are One-based indices
        assert((Si >= 0)&&(Si < NV));
        assert((Ti >= 0)&&(Ti < NV));
        tmpEdgeList[i].head   = Si;  //The S index
        tmpEdgeList[i].tail   = Ti;  //The T index
        tmpEdgeList[i].weight = Wi;   //Make it positive and cast to Double
        i++;
        tmpEdgeList[i].head   = Ti;
        tmpEdgeList[i].tail   = Si;
        tmpEdgeList[i].weight = Wi;
        i++;
    }//End of outer for loop
    fclose(file); //Close the file
    time2 = omp_get_wtime();
    printf("Done reading from file: NE= %ld. Time= %lf\n", NE, time2-time1);
    
    ///////////
    time1 = omp_get_wtime();
    long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
    assert(edgeListPtr != NULL);
    edge *edgeList = (edge *) malloc( (2*NE) * sizeof(edge)); //Every edge stored twice
    assert( edgeList != NULL);
    time2 = omp_get_wtime();
    printf("Time for allocating memory for storing graph = %lf\n", time2 - time1);
    
#pragma omp parallel for
    for (long i=0; i <= NV; i++)
        edgeListPtr[i] = 0; //For first touch purposes
    
    //////Build the EdgeListPtr Array: Cumulative addition
    time1 = omp_get_wtime();
#pragma omp parallel for
    for(long i=0; i<(2*NE); i++) {
        __sync_fetch_and_add(&edgeListPtr[tmpEdgeList[i].head+1], 1); //Leave 0th position intact
    }
    for (long i=0; i<NV; i++) {
        edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
        //printf("%d ",edgeListPtr[i]);
    }
    //The last element of Cumulative will hold the total number of characters
    time2 = omp_get_wtime();
    printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
    printf("Sanity Check: |E| = %ld, edgeListPtr[NV]= %ld\n", 2*NE, edgeListPtr[NV]);
    printf("*********** (%ld)\n", NV);
    
    //time1 = omp_get_wtime();
    //Keep track of how many edges have been added for a vertex:
    printf("About to allocate for added vector: %ld\n", NV);
    long  *added  = (long *)  malloc( NV  * sizeof(long));
    printf("Done allocating memory for added vector\n");
    assert( added != NULL);
#pragma omp parallel for
    for (long i = 0; i < NV; i++)
        added[i] = 0;
    
    printf("About to build edgeList...\n");
    //Build the edgeList from edgeListTmp:
    //#pragma omp parallel for
    for(long i=0; i<(2*NE); i++) {
        long head      = tmpEdgeList[i].head;
        long tail      = tmpEdgeList[i].tail;
        double weight  = tmpEdgeList[i].weight;
        
        long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);
        edgeList[Where].head = head;
        edgeList[Where].tail = tail;
        edgeList[Where].weight = weight;
        
    }
    //time2 = omp_get_wtime();
    printf("Time for building edgeList = %lf\n", time2 - time1);
    
    G->sVertices    = NV;
    G->numVertices  = NV;
    G->numEdges     = NE;
    G->edgeListPtrs = edgeListPtr;
    G->edgeList     = edgeList;
    
    free(tmpEdgeList);
    free(added);
    
}//End of parse_UndirectedEdgeList()


void parse_UndirectedEdgeListFromJason(graph * G, char *fileName) {
    printf("Parsing a SingledEdgeList formatted file as a general graph...\n");
    printf("WARNING: Assumes that the graph is undirected -- an edge is stored ONLY once.\n");
    int nthreads = 0;
    
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    
    double time1, time2;
    
    FILE *file = fopen(fileName, "r");
    if (file == NULL) {
        printf("Cannot open the input file: %s\n",fileName);
        exit(1);
    }
    
    long NV=0, NE=0;
    long nv1, nv2, wt;
    while(!feof(file))
    {
        fscanf(file, "%ld %ld %ld", &nv1, &nv2, &wt);
        if(nv1 > NV)
            NV = nv1;
        if(nv2 > NV)
            NV = nv2;
        NE++;
    }
    NE--;
    //NE*=2;
    fclose(file);
    
    file = fopen(fileName, "r");
    if (file == NULL) {
        printf("Cannot open the input file: %s\n",fileName);
        exit(1);
    }
    char line[1024];
    printf("|V|= %ld, |E|= %ld \n", NV, NE);
    printf("Weights will be converted to positive numbers.\n");
    /*---------------------------------------------------------------------*/
    /* Read edge list: a U V W                                             */
    /*---------------------------------------------------------------------*/
    edge *edgeListTmp = (edge *) malloc(NE * sizeof(edge)); //Every edge stored ONCE
    assert( edgeListTmp != NULL);
    long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
    assert(edgeListPtr != NULL);
    
    long Si, Ti;
    double Twt;
    time1 = omp_get_wtime();
    for (long i = 0; i < NE; i++) {
        fscanf(file, "%ld %ld %lf", &Si, &Ti, &Twt);
        assert((Si >= 0)&&(Si < NV));
        assert((Ti >= 0)&&(Ti < NV));
        edgeListTmp[i].head = Si;       //The S index
        edgeListTmp[i].tail = Ti;    //The T index
        edgeListTmp[i].weight = fabs(Twt); //The value
        edgeListPtr[Si]++;
        edgeListPtr[Ti]++;
    }//End of outer for loop
    fclose(file); //Close the file
    time2 = omp_get_wtime();
    printf("Done reading from file: NE= %ld. Time= %lf\n", NE, time2-time1);
    
    //////Build the EdgeListPtr Array: Cumulative addition
    time1 = omp_get_wtime();
    for (long i=0; i<NV; i++) {
        edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
    }
    //The last element of Cumulative will hold the total number of characters
    time2 = omp_get_wtime();
    printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
    printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld\n", NE*2, edgeListPtr[NV]);
    
    /*---------------------------------------------------------------------*/
    /* Allocate memory for G & Build it                                    */
    /*---------------------------------------------------------------------*/
    time1 = omp_get_wtime();
    edge *edgeList = (edge *) malloc(2 * NE * sizeof(edge)); //Every edge stored twice
    assert(edgeList != 0);
    //Keep track of how many edges have been added for a vertex:
    long  *Counter  = (long *)  malloc(NV  * sizeof(long)); assert(Counter != 0);
#pragma omp parallel for
    for (long i = 0; i < NV; i++)
        Counter[i] = 0;
    time2 = omp_get_wtime();
    printf("Time for allocating memory for marks and edgeList = %lf\n", time2 - time1);
    
    time1 = omp_get_wtime();
    printf("About to build edgeList...\n");
    //Build the edgeList from edgeListTmp:
#pragma omp parallel for
    for(long i=0; i<NE; i++) {
        long head  = edgeListTmp[i].head;
        long tail  = edgeListTmp[i].tail;
        double weight      = edgeListTmp[i].weight;
        if (head != tail) {
            long Where    = edgeListPtr[head] + __sync_fetch_and_add(&Counter[head], 1);
            edgeList[Where].head = head;
            edgeList[Where].tail = tail;
            edgeList[Where].weight = weight;
            //Now add the edge the other way:
            Where                  = edgeListPtr[tail] + __sync_fetch_and_add(&Counter[tail], 1);
            edgeList[Where].head   = tail;
            edgeList[Where].tail   = head;
            edgeList[Where].weight = weight;
        } else {
            long Where    = edgeListPtr[head] + __sync_fetch_and_add(&Counter[head], 1);
            edgeList[Where].head = head;
            edgeList[Where].tail = tail;
            edgeList[Where].weight = weight;
        }
    }
    time2 = omp_get_wtime();
    printf("Time for building edgeList = %lf\n", time2 - time1);
    
    G->sVertices    = NV;
    G->numVertices  = NV;
    G->numEdges     = NE;
    G->edgeListPtrs = edgeListPtr;
    G->edgeList     = edgeList;
    
    free(edgeListTmp);
    free(Counter);
}//End of parse_UndirectedEdgeListFromJason()
