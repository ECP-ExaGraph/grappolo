#include "input_output.h"

void parse_PajekFormat(graph * G, char *fileName) {
  printf("Parsing a Pajek File...\n");
  int nthreads;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();    
  }
  printf("parse_Pajek: Number of threads: %d\n", nthreads);
  
  double time1, time2;
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  //Parse the first line:
  char line[1024];
  fgets(line, 1024, file);  
  char  LS1[25], LS2[25];
  long NV = 0, NE=0;
  if (sscanf(line, "%s %s", LS1, LS2) != 2) {
    printf("parse_Pajek(): bad file format - 01");
    exit(1);
  }  
  //printf("(%s) --- (%s) \n", LS1, LS2);
  if ( strcmp(LS1,"*Vertices")!= 0 ) {
    printf("Error: The first line should start with *Vertices word \n");
    exit(1);
  }
  NV = atol(LS2);
  printf("|V|= %ld \n", NV);
  /* Ignore all the vertex lines */
    do {
        fgets(line, 1024, file);
        sscanf(line, "%s", LS1);
    } while ( strcmp(LS1,"*Edges") != 0 );
    
  //for (long i=0; i <= NV; i++) {
    //fgets(line, 1024, file);
  //}
  printf("Next line:  %s\n", line);
/*
  if (sscanf(line, "%s", LS1) != 1) {
    printf("parse_Pajek(): bad file format - 02");
    exit(1);
  }
  if ( strcmp(LS1,"*Edges")!= 0 ) {
    printf("Error: The next line should start with *Edges word \n");
    exit(1);
  }
  printf("Line read: %s\n",line);
 */
  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /* (i , j, value ) 1-based index                                       */
  /*---------------------------------------------------------------------*/  
  edge *edgeListTmp; //Read the data in a temporary list
  long Si, Ti;
  double weight = 1;
  long edgeEstimate = NV * 1600; //25% density -- not valid for large graphs
  edgeListTmp = (edge *) malloc( edgeEstimate * sizeof(edge));
  assert(edgeListTmp != 0);

  printf("Parsing edges -- with weights\n");
  
  while (fscanf(file, "%ld %ld %lf", &Si, &Ti, &weight) != EOF) {
  //while (fscanf(file, "%ld %ld", &Si, &Ti) != EOF) {
    if(NE >= edgeEstimate) {
      printf("Temporary buffer is not enough. \n");
      exit(1);
    }
    //printf("%ld -- %ld\n", Si, Ti);
    Si--; Ti--;            // One-based indexing
    assert((Si >= 0)&&(Si < NV));
    assert((Ti >= 0)&&(Ti < NV));
    //if(Si % 99998 == 0)
      //printf("%ld -- %ld\n", Si, Ti);
    if (Si == Ti) //Ignore self-loops
      continue; 
    //weight = fabs(weight); //Make it positive    : Leave it as is
    weight = 1.0; //Make it positive    : Leave it as is
    edgeListTmp[NE].head = Si;       //The S index
    edgeListTmp[NE].tail = Ti;    //The T index
    edgeListTmp[NE].weight = weight; //The value
    NE++;
  }
  fclose(file); //Close the file
  printf("Done reading from file.\n");
  printf("|V|= %ld, |E|= %ld \n", NV, NE);
  
  //Remove duplicate entries:
  long NewEdges = removeEdges(NV, NE, edgeListTmp);
  if (NewEdges < NE) {
    printf("Number of duplicate entries detected: %ld\n", NE-NewEdges);
    NE = NewEdges; //Only look at clean edges
  }
  
  //Allocate for Edge Pointer and keep track of degree for each vertex
  long  *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
#pragma omp parallel for
  for (long i=0; i <= NV; i++)
    edgeListPtr[i] = 0; //For first touch purposes

#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    __sync_fetch_and_add(&edgeListPtr[edgeListTmp[i].head + 1], 1); //Plus one to take care of the zeroth location
    __sync_fetch_and_add(&edgeListPtr[edgeListTmp[i].tail + 1], 1);
  }
  
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
  printf("About to allocate memory for graph data structures\n");
  time1 = omp_get_wtime();
  edge *edgeList = (edge *) malloc ((2*NE) * sizeof(edge)); //Every edge stored twice
  assert(edgeList != 0);
  //Keep track of how many edges have been added for a vertex:
  long  *added = (long *)  malloc (NV * sizeof(long)); assert (added != 0);
#pragma omp parallel for
  for (long i = 0; i < NV; i++) 
    added[i] = 0;
  time2 = omp_get_wtime();  
  printf("Time for allocating memory for edgeList = %lf\n", time2 - time1);
  
  time1 = omp_get_wtime();
  
  printf("About to build edgeList...\n");
  //Build the edgeList from edgeListTmp:
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    long head  = edgeListTmp[i].head;
    long tail  = edgeListTmp[i].tail;
    double weight      = edgeListTmp[i].weight;
    
    long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);   
    edgeList[Where].head = head; 
    edgeList[Where].tail = tail;
    edgeList[Where].weight = weight;
    //added[head]++;
    //Now add the counter-edge:
    Where = edgeListPtr[tail] + __sync_fetch_and_add(&added[tail], 1);
    edgeList[Where].head = tail;
    edgeList[Where].tail = head;
    edgeList[Where].weight = weight;
    //added[tail]++;
  }
  time2 = omp_get_wtime();
  printf("Time for building edgeList = %lf\n", time2 - time1);
  
  G->sVertices    = NV;
  G->numVertices  = NV;
  G->numEdges     = NE;
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;
  
  free(edgeListTmp);
  free(added);
}

/*-------------------------------------------------------*
 * This function reads a Pajek file and builds the graph
 *-------------------------------------------------------*/
//Assume every edge is stored twice:
void parse_PajekFormatUndirected(graph * G, char *fileName) {
  printf("Parsing a Pajek File *** Undirected ***...\n");
  int nthreads;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();    
  }
  printf("parse_Pajek_undirected: Number of threads: %d\n", nthreads);
  
  double time1, time2;
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  //Parse the first line:
  char line[1024];
  fgets(line, 1024, file);  
  char  LS1[25], LS2[25];
  long NV = 0, NE=0;
  if (sscanf(line, "%s %s", LS1, LS2) != 2) {
    printf("parse_Pajek(): bad file format - 01");
    exit(1);
  }  
  //printf("(%s) --- (%s) \n", LS1, LS2);
  if ( strcmp(LS1,"*Vertices")!= 0 ) {
    printf("Error: The first line should start with *Vertices word \n");
    exit(1);
  }
  NV = atol(LS2);
  printf("|V|= %ld \n", NV);
    /* Ignore all the vertex lines */
    do {
        fgets(line, 1024, file);
        sscanf(line, "%s", LS1);
    } while ( strcmp(LS1,"*Edges") != 0 );
    printf("Next line:  %s\n", line);
  printf("About to read edges -- no weights\n");
  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /* (i , j, value ) 1-based index                                       */
  /*---------------------------------------------------------------------*/
  
  edge *edgeListTmp; //Read the data in a temporary list
  long Si, Ti;
  double weight = 1;
  long edgeEstimate = NV * NV / 8; //12.5% density -- not valid for dense graphs
  edgeListTmp = (edge *) malloc( edgeEstimate * sizeof(edge));

  //while (fscanf(file, "%ld %ld %lf", &Si, &Ti, &weight) != EOF) {
  while (fscanf(file, "%ld %ld ", &Si, &Ti) != EOF) {
    Si--; Ti--;            // One-based indexing
    assert((Si >= 0)&&(Si < NV));
    assert((Ti >= 0)&&(Ti < NV));
    if (Si == Ti) //Ignore self-loops 
      continue; 
    //weight = fabs(weight); //Make it positive    : Leave it as is
    weight = 1.0; //Make it positive    : Leave it as is
    edgeListTmp[NE].head = Si;       //The S index
    edgeListTmp[NE].tail = Ti;    //The T index
    edgeListTmp[NE].weight = weight; //The value
    NE++;
  }
  fclose(file); //Close the file
  printf("Done reading from file.\n");
  printf("|V|= %ld, |E|= %ld \n", NV, NE);
  
  //Remove duplicate entries:
  /*
  long NewEdges = removeEdges(NV, NE, edgeListTmp);
  if (NewEdges < NE) {
    printf("Number of duplicate entries detected: %ld\n", NE-NewEdges);
    NE = NewEdges; //Only look at clean edges
  } else
    printf("No duplicates were found\n");
  */

  //Allocate for Edge Pointer and keep track of degree for each vertex
  long  *edgeListPtr = (long *) malloc((NV+1) * sizeof(long));
  assert(edgeListPtr != 0);

#pragma omp parallel for
  for (long i=0; i <= NV; i++) {
    edgeListPtr[i] = 0; //For first touch purposes
  }
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    __sync_fetch_and_add(&edgeListPtr[edgeListTmp[i].head + 1], 1); //Plus one to take care of the zeroth location
    //__sync_fetch_and_add(&edgeListPtr[edgeListTmp[i].tail + 1], 1); //No need
  }
  
  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = omp_get_wtime();
  for (long i=0; i<NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = omp_get_wtime();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: |E| = %ld, edgeListPtr[NV]= %ld\n", NE, edgeListPtr[NV]);
  /*---------------------------------------------------------------------*/
  /* Allocate memory for G & Build it                                    */
  /*---------------------------------------------------------------------*/    
  time1 = omp_get_wtime();
  printf("Size of edge: %ld  and size of NE*edge= %ld\n",  sizeof(edge), NE*sizeof(edge) );
  edge *edgeList = (edge *) malloc( NE * sizeof(edge)); //Every edge stored twice
  assert(edgeList != 0);
  //Keep track of how many edges have been added for a vertex:
  long  *added    = (long *)  malloc( NV  * sizeof(long)); assert (added != 0);
 #pragma omp parallel for
  for (long i = 0; i < NV; i++) 
    added[i] = 0;
  
  time2 = omp_get_wtime();
  printf("Time for allocating memory for edgeList = %lf\n", time2 - time1);
  
  time1 = omp_get_wtime();

  printf("About to build edgeList...\n");
  //Build the edgeList from edgeListTmp:
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    long head  = edgeListTmp[i].head;
    long tail  = edgeListTmp[i].tail;
    double weight      = edgeListTmp[i].weight;
    
    long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);   
    edgeList[Where].head = head; 
    edgeList[Where].tail = tail;
    edgeList[Where].weight = weight;
  }
  time2 = omp_get_wtime();
  printf("Time for building edgeList = %lf\n", time2 - time1);
  
  G->sVertices    = NV;
  G->numVertices  = NV;
  G->numEdges     = NE/2; //Each edge had been presented twice
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;
  
  free(edgeListTmp);
  free(added);
}



