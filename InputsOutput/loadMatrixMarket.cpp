#include "input_output.h"
#include "defs.h"
#include "sstream"
#include "utilityStringTokenizer.hpp"


/*-------------------------------------------------------*
 * This function reads a MATRIX MARKET file and builds the graph
 *-------------------------------------------------------*/
//This driver reads in the matrix as follows:
//If symmetric: It creates a graph, where each diagonal entry is a vertex, and
//    each non-diagonal entry becomes an edge. Assume structural and numerical symmetry.
//In unsymetric: It creates a bipartite graph G=(S,T,E). S=row vertices, T=column vertices
//    E=non-zero entries. Vertices are numbered 0 to NS-1 for S vertices, and NS to NS+NT for
//    T vertices.
//WARNING: Weights will be retained as-in (negative weights are not changed)
//Return TRUE if graph symmteric, else FALSE
bool parse_MatrixMarket(graph * G, char *fileName) {
    printf("Parsing a Matrix Market File...\n");
    int nthreads;
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    printf("parse_MatrixMarket: Number of threads: %d\n", nthreads);
    
    double time1, time2;
    FILE *file = fopen(fileName, "r");
    if (file == NULL) {
        printf("Cannot open the input file: %s\n",fileName);
        exit(1);
    }
    
    /* -----      Read File in Matrix Market Format     ------ */
    //Parse the first line:
    char line[1024];
    fgets(line, 1024, file);
    char  LS1[25], LS2[25], LS3[25], LS4[25], LS5[25];
    if (sscanf(line, "%s %s %s %s %s", LS1, LS2, LS3, LS4, LS5) != 5) {
        printf("parse_MatrixMarket(): bad file format - 01");
        exit(1);
    }
    printf("%s %s %s %s %s\n", LS1, LS2, LS3, LS4, LS5);
    if ( strcmp(LS1,"%%MatrixMarket") != 0 ) {
        printf("Error: The first line should start with %%MatrixMarket word \n");
        exit(1);
    }
    if ( !( strcmp(LS2,"matrix")==0 || strcmp(LS2,"Matrix")==0 || strcmp(LS2,"MATRIX")==0 ) ) {
        printf("Error: The Object should be matrix or Matrix or MATRIX \n");
        exit(1);
    }
    if ( !( strcmp(LS3,"coordinate")==0 || strcmp(LS3,"Coordinate")==0 || strcmp(LS3,"COORDINATE")==0) ) {
        printf("Error: The Object should be coordinate or Coordinate or COORDINATE \n");
        exit(1);
    }
    int isComplex = 0;
    if ( strcmp(LS4,"complex")==0 || strcmp(LS4,"Complex")==0 || strcmp(LS4,"COMPLEX")==0 ) {
        isComplex = 1;
        printf("Warning: Will only read the real part. \n");
    }
    int isPattern = 0;
    if ( strcmp(LS4,"pattern")==0 || strcmp(LS4,"Pattern")==0 || strcmp(LS4,"PATTERN")==0 ) {
        isPattern = 1;
        printf("Note: Matrix type is Pattern. Will set all weights to 1.\n");
        //exit(1);
    }
    int isSymmetric = 0, isGeneral = 0;
    if ( strcmp(LS5,"general")==0 || strcmp(LS5,"General")==0 || strcmp(LS5,"GENERAL")==0 )
        isGeneral = 1;
    else {
        if ( strcmp(LS5,"symmetric")==0 || strcmp(LS5,"Symmetric")==0 || strcmp(LS5,"SYMMETRIC")==0 ) {
            isSymmetric = 1;
            printf("Note: Matrix type is Symmetric: Converting it into General type. \n");
        }
    }
    if ( (isGeneral==0) && (isSymmetric==0) ) 	  {
        printf("Warning: Matrix type should be General or Symmetric. \n");
        exit(1);
    }
    
    /* Parse all comments starting with '%' symbol */
    do {
        fgets(line, 1024, file);
    } while ( line[0] == '%' );
    
    /* Read the matrix parameters */
    long NS=0, NT=0, NV = 0;
    long NE=0;
    if (sscanf(line, "%ld %ld %ld", &NS, &NT, &NE ) != 3) {
        printf("parse_MatrixMarket(): bad file format - 02");
        exit(1);
    }
    if (isSymmetric == 1) {
        assert(NS == NT);
        NV = NS;
    } else {
        NV = NS + NT;
    }
    printf("|S|= %ld, |T|= %ld, |E|= %ld \n", NS, NT, NE);
    
    /*---------------------------------------------------------------------*/
    /* Read edge list                                                      */
    /* S vertices: 0 to NS-1                                               */
    /* T vertices: NS to NS+NT-1                                           */
    /*---------------------------------------------------------------------*/
    //Allocate for Edge Pointer and keep track of degree for each vertex
    long  *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
#pragma omp parallel for
    for (long i=0; i <= NV; i++)
        edgeListPtr[i] = 0; //For first touch purposes
    
    edge *edgeListTmp; //Read the data in a temporary list
    long newNNZ = 0;    //New edges because of symmetric matrices
    long Si, Ti;
    double weight = 1;
    if( isSymmetric == 1 ) {
        printf("Matrix is of type: Symmetric Real or Complex\n");
        //printf("Weights will be converted to positive numbers.\n");
        printf("Weights will be retianed as-is.\n");
        edgeListTmp = (edge *) malloc(2 * NE * sizeof(edge));
        for (long i = 0; i < NE; i++) {
            if (isPattern == 1)
                fscanf(file, "%ld %ld", &Si, &Ti);
            else
                fscanf(file, "%ld %ld %lf", &Si, &Ti, &weight);
            Si--; Ti--;            // One-based indexing
            assert((Si >= 0)&&(Si < NV));
            assert((Ti >= 0)&&(Ti < NV));
            //weight = fabs(weight); //Make it positive  : Leave it as is
            if ( Si == Ti ) { //Add a self loop:
                edgeListTmp[newNNZ].head = Si;  //The S index
                edgeListTmp[newNNZ].tail = Si;  //The T index
                edgeListTmp[newNNZ].weight = weight; //The value
                edgeListPtr[Si+1]++;
                newNNZ++; //Increment the number of edges
            }
            else { //an off diagonal element: Store the edge
                edgeListTmp[newNNZ].head = Si;       //The S index
                edgeListTmp[newNNZ].tail = Ti;    //The T index
                edgeListTmp[newNNZ].weight = weight; //The value
                edgeListPtr[Ti+1]++;
                edgeListPtr[Si+1]++;
                newNNZ++; //Increment the number of edges
            }
        }
        printf("Modified the number of edges from %ld ",NE);
        NE = newNNZ; //#NNZ might change
        printf("to %ld \n",NE);
    } //End of Symmetric
    /////// General Real or Complex ///////
    else {
        printf("Matrix is of type: Unsymmetric Real or Complex\n");
        //printf("Weights will be converted to positive numbers.\n");
        printf("Weights will be retained as-is.\n");
        edgeListTmp = (edge *) malloc( NE * sizeof(edge));
        for (long i = 0; i < NE; i++) {
            if (isPattern == 1)
                fscanf(file, "%ld %ld", &Si, &Ti);
            else
                fscanf(file, "%ld %ld %lf", &Si, &Ti, &weight);
            //printf("(%d, %d) %lf\n",Si, Ti, weight);
            Si--; Ti--;            // One-based indexing
            assert((Si >= 0)&&(Si < NV));
            assert((Ti >= 0)&&(Ti < NV));
            //weight = fabs(weight); //Make it positive    : Leave it as is
            edgeListTmp[i].head = Si;       //The S index
            edgeListTmp[i].tail = NS+Ti;    //The T index
            edgeListTmp[i].weight = weight; //The value
            edgeListPtr[Si+1]++;
            edgeListPtr[NS+Ti+1]++;
        }
    } //End of Real or Complex
    fclose(file); //Close the file
    printf("Done reading from file.\n");
    
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
    edge *edgeList = (edge *) malloc( 2*NE * sizeof(edge)); //Every edge stored twice
    assert(edgeList != 0);
    //Keep track of how many edges have been added for a vertex:
    long  *Counter    = (long *)  malloc( NV  * sizeof(long)); assert(Counter != 0);
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
    
    G->sVertices    = NS;
    G->numVertices  = NV;
    G->numEdges     = NE;
    G->edgeListPtrs = edgeListPtr;
    G->edgeList     = edgeList;
    
    free(edgeListTmp);
    free(Counter);
    
    if (isSymmetric == 1)
        return true;
    else
        return false;
} //End of parse_MatrixMarket()


/*-------------------------------------------------------*
 * This function reads a MATRIX MARKET file and build the graph
 * graph is nonbipartite: each diagonal entry is a vertex, and
 * each non-diagonal entry becomes an edge. Assume structural and
 * numerical symmetry.
 *-------------------------------------------------------*/
void parse_MatrixMarket_Sym_AsGraph(graph * G, char *fileName) {
    printf("Parsing a Matrix Market File as a general graph...\n");
    int nthreads = 0;
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    printf("parse_MatrixMarket: Number of threads: %d\n ", nthreads);
    
    double time1, time2;
    FILE *file = fopen(fileName, "r");
    if (file == NULL) {
        printf("Cannot open the input file: %s\n",fileName);
        exit(1);
    }
    /* -----      Read File in Matrix Market Format     ------ */
    //Parse the first line:
    char line[1024];
    fgets(line, 1024, file);
    char  LS1[25], LS2[25], LS3[25], LS4[25], LS5[25];
    if (sscanf(line, "%s %s %s %s %s", LS1, LS2, LS3, LS4, LS5) != 5) {
        printf("parse_MatrixMarket(): bad file format - 01");
        exit(1);
    }
    printf("%s %s %s %s %s\n", LS1, LS2, LS3, LS4, LS5);
    if ( strcmp(LS1,"%%MatrixMarket") != 0 ) {
        printf("Error: The first line should start with %%MatrixMarket word \n");
        exit(1);
    }
    if ( !( strcmp(LS2,"matrix")==0 || strcmp(LS2,"Matrix")==0 || strcmp(LS2,"MATRIX")==0 ) ) {
        printf("Error: The Object should be matrix or Matrix or MATRIX \n");
        exit(1);
    }
    if ( !( strcmp(LS3,"coordinate")==0 || strcmp(LS3,"Coordinate")==0 || strcmp(LS3,"COORDINATE")==0) ) {
        printf("Error: The Object should be coordinate or Coordinate or COORDINATE \n");
        exit(1);
    }
    int isComplex = 0;
    if ( strcmp(LS4,"complex")==0 || strcmp(LS4,"Complex")==0 || strcmp(LS4,"COMPLEX")==0 ) {
        isComplex = 1;
        printf("Warning: Will only read the real part. \n");
    }
    int isPattern = 0;
    if ( strcmp(LS4,"pattern")==0 || strcmp(LS4,"Pattern")==0 || strcmp(LS4,"PATTERN")==0 ) {
        isPattern = 1;
        printf("Note: Matrix type is Pattern. Will set all weights to 1.\n");
        //exit(1);
    }
    int isSymmetric = 0, isGeneral = 0;
    if ( strcmp(LS5,"general")==0 || strcmp(LS5,"General")==0 || strcmp(LS5,"GENERAL")==0 )
        isGeneral = 1;
    else {
        if ( strcmp(LS5,"symmetric")==0 || strcmp(LS5,"Symmetric")==0 || strcmp(LS5,"SYMMETRIC")==0 ) {
            isSymmetric = 1;
            printf("Note: Matrix type is Symmetric: Converting it into General type. \n");
        }
    }
    if ( isSymmetric==0 )       {
        printf("Warning: Matrix type should be Symmetric for this routine. \n");
        printf("Warning: Matrix will be considered as a symmetric matrix -- only entries in the lower triangle will be considered. \n");
        printf("Warning: .... Diagonal entries (self loops) will be ignored. \n");
    } else {
        printf("Matrix is of type: Symmetric Real or Complex\n");
        printf("Warning: .... Diagonal entries (self loops) will be ignored. \n");
    }
    
    /* Parse all comments starting with '%' symbol */
    do {
        fgets(line, 1024, file);
    } while ( line[0] == '%' );
    
    /* Read the matrix parameters */
    long NS=0, NT=0, NV = 0;
    long NE=0;
    if (sscanf(line, "%ld %ld %ld",&NS, &NT, &NE ) != 3) {
        printf("parse_MatrixMarket(): bad file format - 02");
        exit(1);
    }
    NV = NS;
    printf("|S|= %ld, |T|= %ld, |E|= %ld \n", NS, NT, NE);
    
    /*---------------------------------------------------------------------*/
    /* Read edge list                                                      */
    /* S vertices: 0 to NS-1                                               */
    /* T vertices: NS to NS+NT-1                                           */
    /*---------------------------------------------------------------------*/
    //Allocate for Edge Pointer and keep track of degree for each vertex
    long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
#pragma omp parallel for
    for (long i=0; i <= NV; i++)
        edgeListPtr[i] = 0; //For first touch purposes
    
    edge *edgeListTmp; //Read the data in a temporary list
    long newNNZ = 0;    //New edges because of symmetric matrices
    long Si, Ti;
    double weight = 1;
    printf("Weights will be converted to positive numbers.\n");
    edgeListTmp = (edge *) malloc(NE * sizeof(edge));
    while ( !feof(file)  ) {
        if (isPattern == 1)
            fscanf(file, "%ld %ld", &Si, &Ti);
        else
            fscanf(file, "%ld %ld %lf", &Si, &Ti, &weight);
        if (Ti >= Si)
            continue; //Ignore the upper triangluar part of the matrix
        Si--; Ti--;            // One-based indexing
        assert((Si >= 0)&&(Si < NV));
        assert((Ti >= 0)&&(Ti < NV));
        weight = fabs(weight); //Make it positive  : Leave it as is
        //LOWER PART:
        edgeListTmp[newNNZ].head = Si;       //The S index
        edgeListTmp[newNNZ].tail = Ti;       //The T index
        edgeListTmp[newNNZ].weight = weight; //The value
        edgeListPtr[Si+1]++;
        edgeListPtr[Ti+1]++;
        newNNZ++;
    }//End of while loop
    fclose(file); //Close the file
    //newNNZ = newNNZ / 2;
    printf("Done reading from file.\n");
    printf("Modified the number of edges from %ld ", NE);
    NE = newNNZ; //#NNZ might change
    printf("to %ld \n", NE);
    
    //////Build the EdgeListPtr Array: Cumulative addition
    time1 = omp_get_wtime();
    for (long i=0; i<NV; i++) {
        edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
    }
    //The last element of Cumulative will hold the total number of characters
    time2 = omp_get_wtime();
    printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
    assert(NE*2 ==  edgeListPtr[NV]);
    //printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld\n", NE*2, edgeListPtr[NV]);
    
    /*---------------------------------------------------------------------*/
    /* Allocate memory for G & Build it                                    */
    /*---------------------------------------------------------------------*/
    time1 = omp_get_wtime();
    edge *edgeList = (edge *) malloc( 2*NE * sizeof(edge)); //Every edge stored twice
    assert(edgeList != 0);
    //Keep track of how many edges have been added for a vertex:
    long  *Counter = (long *) malloc (NV  * sizeof(long)); assert(Counter != 0);
#pragma omp parallel for
    for (long i = 0; i < NV; i++) {
        Counter[i] = 0;
    }
    time2 = omp_get_wtime();
    printf("Time for allocating memory for edgeList = %lf\n", time2 - time1);
    printf("About to build edgeList...\n");
    
    time1 = omp_get_wtime();
    //Build the edgeList from edgeListTmp:
#pragma omp parallel for
    for(long i=0; i<NE; i++) {
        long head     = edgeListTmp[i].head;
        long tail     = edgeListTmp[i].tail;
        double weight = edgeListTmp[i].weight;
        long Where    = edgeListPtr[head] + __sync_fetch_and_add(&Counter[head], 1);
        edgeList[Where].head = head;
        edgeList[Where].tail = tail;
        edgeList[Where].weight = weight;
        //Now add the edge the other way:
        Where                  = edgeListPtr[tail] + __sync_fetch_and_add(&Counter[tail], 1);
        edgeList[Where].head   = tail;
        edgeList[Where].tail   = head;
        edgeList[Where].weight = weight;
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
    
}//End of parse_MatrixMarket_Sym_AsGraph()
