#include "input_output.h"
#include "utilityStringTokenizer.hpp"

/* ****************************************** */
//Loading Functions:
/* NOTE: Indices are ZERO-based, i.e. G(0,0) is the first element,
 while the indices stored in the file are ONE-based.
 Details:
 * A graph contains n nodes and m arcs
 * Nodes are identified by integers 1...* Graphs can be interpreted as directed or undirected, depending on the problem being studied
 * Graphs can have parallel arcs and self-loops
 * Arc weights are signed integers
 
 ** #... : This is a comment
 ** # Nodes: 65608366 Edges: 1806067135
 ** U V   : U = from; V = to -- is an edge
 
 * Assumption: Each edge is stored ONLY ONCE.
 */
void parse_SNAP(graph * G, char *fileName) {
    printf("Parsing a SNAP formatted file as a general graph...\n");
    printf("WARNING: Assumes that the graph is directed -- an edge is stored only once.\n");
    printf("       : Graph will be stored as undirected, each edge appears twice.\n");
    int nthreads = 0;
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    printf("parse_SNAP: Number of threads: %d\n ", nthreads);
    
    long   NV=0,  NE=0;
    string oneLine, myDelimiter(" "), myDelimiter2("\t"), oneWord; //Delimiter is a blank space
    char comment;
    
    double time1, time2;
    ifstream fin;
    fin.open(fileName);
    if(!fin) {
        cerr<<"Within Function: loadSNAPFileFormat() \n";
        cerr<<"Could not open the file.. \n";
        exit(1);
    }
    
    do { //Parse the comment lines for problem size
        getline(fin, oneLine);
        cout<<"Read line: "<<oneLine<<endl;
        comment = oneLine[0];
        if (comment == '#') { //Check if this line has problem sizes
            StringTokenizer* ST = new StringTokenizer(oneLine, myDelimiter);
            if ( ST->HasMoreTokens() )
                oneWord = ST->GetNextToken(); //Ignore #
            if ( ST->HasMoreTokens() )
                oneWord = ST->GetNextToken(); //Ignore #
            if(oneWord == "Nodes:") {
                NV  = atol( ST->GetNextToken().c_str() ); //Number of Vertices
                oneWord = ST->GetNextToken(); //Ignore Edges:
                NE  = atol( ST->GetNextToken().c_str() ); //Number of Edges
            }
            delete ST;
        }
    } while ( comment == '#');
    
    printf("|V|= %ld, |E|= %ld \n", NV, NE);
    printf("Weights will read from the file.\n");
    cout << oneLine <<endl;
    /*---------------------------------------------------------------------*/
    /* Read edge list: a U V W                                             */
    /*---------------------------------------------------------------------*/
    edge *tmpEdgeList = (edge *) malloc( NE * sizeof(edge)); //Every edge stored ONCE
    assert( tmpEdgeList != NULL);
    long Si, Ti;
    
    map<long, long> clusterLocalMap; //Renumber vertices contiguously from zero
    map<long, long>::iterator storedAlready;
    long numUniqueVertices = 0;
    
    //Parse the first edge already read from the file and stored in oneLine
    long i=0;
    double wt = 1.0;
    do {
        StringTokenizer* ST = new StringTokenizer(oneLine, myDelimiter2);
        if ( ST->HasMoreTokens() )
            Si  = atol( ST->GetNextToken().c_str() );
        if ( ST->HasMoreTokens() )
            Ti  = atol( ST->GetNextToken().c_str() );
        if ( ST->HasMoreTokens() )
            wt  = atof( ST->GetNextToken().c_str() );
        delete ST;
        
        storedAlready = clusterLocalMap.find(Si); //Check if it already exists
        if( storedAlready != clusterLocalMap.end() ) {	//Already exists
            Si = storedAlready->second; //Renumber the cluster id
        } else {
            clusterLocalMap[Si] = numUniqueVertices; //Does not exist, add to the map
            Si = numUniqueVertices; //Renumber the vertex id
            numUniqueVertices++; //Increment the number
        }
        
        storedAlready = clusterLocalMap.find(Ti); //Check if it already exists
        if( storedAlready != clusterLocalMap.end() ) {	//Already exists
            Ti = storedAlready->second; //Renumber the cluster id
        } else {
            clusterLocalMap[Ti] = numUniqueVertices; //Does not exist, add to the map
            Ti = numUniqueVertices; //Renumber the vertex id
            numUniqueVertices++; //Increment the number
        }
        tmpEdgeList[i].head   = Si;  //The S index
        tmpEdgeList[i].tail   = Ti;  //The T index: One-based indexing
        tmpEdgeList[i].weight = wt;     //default weight of one
        //cout<<" Adding edge ("<<Si<<", "<<Ti<<")\n";
        i++;
        //Read-in the next line
        getline(fin, oneLine);
        if ((i % 99999) == 1) {
            cout <<"Reading Line: "<<i<<endl;
        }
    } while ( !fin.eof() );//End of while
    
    fin.close(); //Close the file
    time2 = omp_get_wtime();
    printf("Done reading from file: NE= %ld. Time= %lf\n", NE, time2-time1);
    printf("Number of unique vertices: %ld \n", numUniqueVertices);
    
    NV = numUniqueVertices;
    ///////////
    time1 = omp_get_wtime();
    long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
    assert(edgeListPtr != NULL);
    edge *edgeList = (edge *) malloc( 2*NE * sizeof(edge)); //Every edge stored twice
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
        __sync_fetch_and_add(&edgeListPtr[tmpEdgeList[i].tail+1], 1);
    }
    for (long i=0; i<NV; i++) {
        edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
    }
    //The last element of Cumulative will hold the total number of characters
    time2 = omp_get_wtime();
    printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
    printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld\n", NE*2, edgeListPtr[NV]);
    printf("*********** (%ld)\n", NV);
    
    printf("About to build edgeList...\n");
    time1 = omp_get_wtime();
    //Keep track of how many edges have been added for a vertex:
    long  *added  = (long *)  malloc( NV  * sizeof(long)); assert( added != NULL);
#pragma omp parallel for
    for (long i = 0; i < NV; i++)
        added[i] = 0;
    printf("...\n");
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
        //Now add the counter-edge:
        Where = edgeListPtr[tail] + __sync_fetch_and_add(&added[tail], 1);
        edgeList[Where].head = tail;
        edgeList[Where].tail = head;
        edgeList[Where].weight = weight;    
    }
    time2 = omp_get_wtime();
    printf("Time for building edgeList = %lf\n", time2 - time1);
    
    ///////Store the vertex ids in a file////////
    char filename2[256];
    sprintf(filename2,"%s_vertexMap.txt", fileName);
    printf("Writing vertex map (new id -- old id) in file: %s\n", filename2);
    FILE *fout;
    fout = fopen(filename2, "w");
    if (!fout) {
        printf("Could not open the file \n");
        exit(1);
    }
    //Write the edges (lower triangle only):
    storedAlready = clusterLocalMap.begin();
    while (storedAlready != clusterLocalMap.end()) {
        fprintf(fout, "%ld %ld\n", storedAlready->second, storedAlready->first);
        storedAlready++;
    }
    fclose(fout);
    printf("Vertex map has been stored in file: %s\n",filename2);
    
    G->sVertices    = NV;
    G->numVertices  = NV;
    G->numEdges     = NE;
    G->edgeListPtrs = edgeListPtr;
    G->edgeList     = edgeList;
    
    //Clean up
    free(tmpEdgeList);
    free(added);
}//End of parse_SNAP()

/* Parse files with ground truth information. Needs two files:
 1. Vertex map -- format: (new-id  old-id) on each line
 2. Ground truth information: Each line lists the vertex ids for a given community
*/
void parse_SNAP_GroundTruthCommunities(char *fileVertexMap, char *fileGroundTruth) {
    printf("Within parse_SNAP_GroundTruthCommunities()\n");
    int nthreads = 0;
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    printf("parse_SNAP_GroundTruthCommunities: Number of threads: %d\n ", nthreads);
    
    long   NV=0,  NC=0;
    string oneLine, myDelimiter(" "), myDelimiter2("\t"), oneWord; //Delimiter is a blank space
    char comment;
    
    double time1, time2;
    ifstream fin;
    fin.open(fileVertexMap);
    if(!fin) {
        cerr<<"Within Function: parse_SNAP_GroundTruthCommunities() \n";
        cerr<<"Could not open the file.. \n";
        exit(1);
    }
    
    //Parse the vertex id mapping:  new-id --> old-id
    map<long, long> clusterLocalMap; //Renumber vertices contiguously from zero
    map<long, long>::iterator storedAlready;
    long Si, Ti;
    printf("Parsing file %s (new-id  old-id)...\n ", fileVertexMap);
    do {
        StringTokenizer* ST = new StringTokenizer(oneLine, myDelimiter);
        if ( ST->HasMoreTokens() )
            Si  = atol( ST->GetNextToken().c_str() ); //New id -- value
        if ( ST->HasMoreTokens() )
            Ti  = atol( ST->GetNextToken().c_str() ); //Old id -- key
        delete ST;
        
        printf("%ld \t\t %ld\n", Si, Ti);
        clusterLocalMap[Ti] = Si; //Does not exist, add to the map
        NV++; //Increment the number
        
        /* storedAlready = clusterLocalMap.find(Ti); //Check if it already exists
        if( storedAlready != clusterLocalMap.end() ) {	//Already exists
            printf("WARNING -- vertex id repeated. Something wrong\n\n"); //Renumber the cluster id
        } else {
            clusterLocalMap[Ti] = Si; //Does not exist, add to the map
            NV++; //Increment the number
        } */
        
        //Read-in the next line
        getline(fin, oneLine);
    } while ( !fin.eof() );//End of while
    
    fin.close();
    printf("Finished parsing vertex mapping. |V|= %ld \n\n", NV);
    
    //Parse the ground-truth community file
    fin.open(fileGroundTruth);
    if(!fin) {
        cerr<<"Within Function: parse_SNAP_GroundTruthCommunities() \n";
        cerr<<"Could not open the file.. \n";
        exit(1);
    }
    
    //Parse the vertex id mapping:  new-id --> old-id
    long * communityMap = (long *) malloc (NV * sizeof(long)); assert(communityMap != 0);
    for(long i=0; i<NV; i++)
        communityMap[i] = -1;
    Si = 0;
    printf("Parsing file %s -- assumes that the new vertex ids are in an order\n ", fileGroundTruth);
    do {
        StringTokenizer* ST = new StringTokenizer(oneLine, myDelimiter2);
        while( ST->HasMoreTokens() ) {
            Ti = atol(ST->GetNextToken().c_str());
            storedAlready = clusterLocalMap.find(Ti);
            long where = storedAlready->second;
            printf("%ld \t\t %ld\n", Ti, where);
            assert(where < NV);
            communityMap[where] = Si;
        }
        delete ST; //Clear the buffer
        Si++;
        //Read-in the next line
        getline(fin, oneLine);
        if ((Si % 99999) == 1) {
            cout <<"Reading Line: "<<Si<<endl;
        }
    } while ( !fin.eof() );//End of while
    fin.close();
    printf("Finished parsing ground truth information. |C|= %ld \n\n", Si);
    
    ///////Store the ground truth information in a file////////
    char filename2[256];
    sprintf(filename2,"%s_GroundTruthNew.txt", fileGroundTruth);
    printf("Writing new ground truth information in file: %s\n", filename2);
    FILE *fout;
    fout = fopen(filename2, "w");
    if (!fout) {
        printf("Could not open the file \n");
        exit(1);
    }
    //Write the communities for each vertex:
    for(long i=0; i<NV; i++) {
        fprintf(fout, "%ld %ld\n", i+1, communityMap[i]);
    }
    fclose(fout);
  
    //Clean up
    free(communityMap);
    
}//End of parse_SNAP_GroundTruthCommunities()
