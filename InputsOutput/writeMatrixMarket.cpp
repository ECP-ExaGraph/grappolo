#include "input_output.h"
void writeGraphMatrixMarketFormatSymmetric(graph* G, char *filename) {
    //Get the iterators for the graph:
    long NVer     = G->numVertices;
    long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
    long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
    edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
    printf("NVer= %ld --  NE=%ld\n", NVer, NEdge);
    
    printf("Writing graph in Matrix Market (symmetric) format - each edge represented ONLY ONCE!\n");
    printf("Matrix will be stored in file: %s\n", filename);
    
    FILE *fout;
    fout = fopen(filename, "w");
    if (!fout) {
        printf("Could not open the file \n");
        exit(1);
    }
    //First Line: Header for Matrix Market:
    fprintf(fout, "\%\%MatrixMarket matrix coordinate real symmetric\n");
    fprintf(fout, "\%=================================================================================\n");
    fprintf(fout, "\% Indices are 1-based, i.e. A(1,1) is the first element.\n");
    fprintf(fout, "\% Can contain self-loops (diagonal entries)\n");
    fprintf(fout, "\% Number of edges might not match with actual edges (nonzeros).\n");
    fprintf(fout, "\%=================================================================================\n");
    fprintf(fout, "%ld %ld %ld\n", NVer, NVer, NEdge);
    
    //Write the edges (lower triangle only):
    for (long v=0; v<NVer; v++) {
        long adj1 = verPtr[v];
        long adj2 = verPtr[v+1];
        //Edge lines: <adjacent> <weight>
        for(long k = adj1; k < adj2; k++ ) {
            if (verInd[k].tail <= v ) { //Print only once (lower triangle)
                fprintf(fout, "%ld %ld %g\n", v+1, (verInd[k].tail+1), (verInd[k].weight) );
            }
        }
    }
    fclose(fout);
    printf("Matrix has been stored in file: %s\n",filename);
}//End of writeGraphPajekFormat()


//This routine outputs the graph as a "reordered" symmetric matrix
//The vector old2NewMap contains the new ids for old ids
void writeGraphMatrixMarketFormatSymmetricReordered(graph* G, char *filename, long *old2NewMap) {
    printf("Within writeGraphMatrixMarketFormatSymmetricReordered()\n");
    //Get the iterators for the graph:
    long NVer     = G->numVertices;
    long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
    long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
    edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
    printf("NVer= %ld --  NE=%ld\n", NVer, NEdge);
    
    printf("Writing graph in Matrix Market (symmetric) format - each edge represented ONLY ONCE!\n");
    printf("Matrix will be stored in file: %s\n", filename);
    
    FILE *fout;
    fout = fopen(filename, "w");
    if (!fout) {
        printf("Could not open the file \n");
        exit(1);
    }
    //First Line: Header for Matrix Market:
    fprintf(fout, "\%\%\%\%MatrixMarket matrix coordinate real symmetric\n");
    fprintf(fout, "\%\%=================================================================================\n");
    fprintf(fout, "\%\% Indices are 1-based, i.e. A(1,1) is the first element.\n");
    fprintf(fout, "\%\% Can contain self-loops (diagonal entries)\n");
    fprintf(fout, "\%\% Number of edges might not match with actual edges (nonzeros).\n");
    fprintf(fout, "\%\%=================================================================================\n");
    fprintf(fout, "%ld %ld %ld\n", NVer, NVer, NEdge);
    
    //Write the edges
    for (long v=0; v<NVer; v++) {
        long adj1 = verPtr[v];
        long adj2 = verPtr[v+1];
        //Edge lines: <adjacent> <weight>
        for(long k = adj1; k < adj2; k++ ) {
            if (verInd[k].tail <= v ) { //Print only once (lower triangle)
                fprintf(fout, "%ld %ld %g\n", old2NewMap[v]+1, (old2NewMap[verInd[k].tail]+1), (verInd[k].weight) );
            }
        }
    }
    fclose(fout);
    printf("Matrix has been stored in file: %s\n",filename);
}//End of writeGraphPajekFormat()

//This routine outputs the graph as a "reordered" symmetric matrix
//The vector old2NewMap contains the new ids for old ids
void writeGraphMatrixMarketFormatBipartiteReordered(graph* G, char *filename, long *old2NewMap) {
    printf("Within writeGraphMatrixMarketFormatBipartiteReordered()\n");
    //Get the iterators for the graph:
    long NVer     = G->numVertices;
    long NS       = G->sVertices;
    long NT       = NVer - NS;  assert(NT > 0); //Make sure that the graph is bipartite
    long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
    long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
    edge *verInd  = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
    printf("NS= %ld -- NT= %ld --  NE=%ld\n", NS, NT, NEdge);
    
    printf("Writing graph in Matrix Market (unsymmetric) format - each edge represented ONCE!\n");
    printf("Matrix will be stored in file: %s\n", filename);
    
    FILE *fout;
    fout = fopen(filename, "w");
    if (!fout) {
        printf("Could not open the file \n");
        exit(1);
    }
    
    //First Line: Header for Matrix Market:
    fprintf(fout, "\%\%\%\%MatrixMarket matrix coordinate real general\n");
    fprintf(fout, "\%\%=================================================================================\n");
    fprintf(fout, "\%\% Indices are 1-based, i.e. A(1,1) is the first element.\n");
    fprintf(fout, "\%\% Edges (nonzeros) appear only once.\n");
    fprintf(fout, "\%\%=================================================================================\n");
    fprintf(fout, "%ld %ld %ld\n", NS, NT, NEdge);
    //Write the edges (Only from the row perspective):
    for (long v=0; v<NS; v++) {
        long adj1 = verPtr[v];
        long adj2 = verPtr[v+1];
        //Edge lines: <adjacent> <weight>
        for(long k = adj1; k < adj2; k++ ) {
            fprintf(fout, "%ld %ld %g\n", old2NewMap[v]+1, old2NewMap[verInd[k].tail]-NS+1, verInd[k].weight );
        }//End of for(k)
    }//End of for(v)
    fclose(fout);
    printf("Matrix has been stored in file: %s\n",filename);
}//End of writeGraphPajekFormat()

