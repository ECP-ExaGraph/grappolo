#include "input_output.h"
void writeGraphBinaryFormatNew(graph* G, char *filename, long weighted) {
  //Get the iterators for the graph:
  long NVer    = G->numVertices;
  long NEdge   = G->numEdges;       //Returns the correct number of edges (not twice)
  long *verPtr = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
  printf("NVer= %ld --  NE=%ld\n", NVer, NEdge);
  printf("Writing graph in a simple edgelist format - each edge stored TWICE.\n");
 
  ofstream ofs;
  ofs.open(filename, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
  if (!ofs) {
    std::cerr << "Error opening output file: " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  //First Line: #Vertices #Edges
  ofs.write(reinterpret_cast<char*>(&NVer), sizeof(NVer));
  ofs.write(reinterpret_cast<char*>(&NEdge), sizeof(NEdge));
  ofs.write(reinterpret_cast<char*>(&weighted),sizeof(weighted));

  //Write all the edges (each edge stored twice):
  ofs.write(reinterpret_cast<char*>(verPtr), (NVer+1)*sizeof(long));
  ofs.write(reinterpret_cast<char*>(verInd), (2*NEdge)*sizeof(edge));

  ofs.close();
  printf("Graph has been stored in file: %s\n",filename);
}//End of writeGraphBinaryFormatTwice()