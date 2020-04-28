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

int main(int argc, char** argv) {

   //Parse Input parameters:
  clustering_parameters opts;
  if (!opts.parse(argc, argv)) {
    return -1;
  }
  int nT = 1; //Default is one thread
#pragma omp parallel
  {
      nT = omp_get_num_threads();
  }
  if (nT <= 1) {
	printf("The number of threads should be greater than one.\n");
	return 0;
  }
  graph* G = (graph *) malloc (sizeof(graph));

  int fType = opts.ftype; //File type
  char *inFile = (char*) opts.inFile;
 
  if(fType == 1)
     parse_MatrixMarket_Sym_AsGraph(G, inFile);
  else if(fType == 2)
     parse_Dimacs9FormatDirectedNewD(G, inFile);
  else if(fType == 3)
     parse_PajekFormat(G, inFile);
  else if(fType == 4)
     parse_PajekFormatUndirected(G, inFile);
  else if(fType == 5)
     loadMetisFileFormat(G, inFile); 
  else if(fType == 6)
     parse_DoulbedEdgeList(G, inFile); 
  else
     parse_EdgeListBinary(G, inFile);
  
  displayGraphCharacteristics(G);

  long NVer = G->numVertices;
  //Vector for storing colors
  int *colors = (int *) malloc (NVer * sizeof(int)); assert (colors != 0);
#pragma omp parallel for
  for (long i=0; i<NVer; i++) {
	colors[i] = -1;
  }
  double tmpTime;
  int numColors = algoDistanceOneVertexColoringOpt(G, colors, nT, &tmpTime);
  printf("Time to color: %lf\n", tmpTime);
 	
  //Compute Gini Coefficient:
  long *colorFreq = (long *) malloc (numColors * sizeof(long)); assert(colorFreq != 0);
#pragma omp parallel for
  for(long i = 0; i < numColors; i++) {
       colorFreq[i] = 0;
  }

  buildColorSize(NVer, colors, numColors, colorFreq);
  printf("Done building the colorsize\n");
  computeVariance(NVer, numColors, colorFreq);

  //Compute Gini coefficient    
  double GiniCoeff = computeGiniCoefficient(colorFreq, numColors);
  printf("************************************************\n");
  printf("Gini coefficient: %g\n", GiniCoeff);
  printf("************************************************\n");
  
  /*
    char buf1[256];
    sprintf(buf1,"%s_GiniPlot.txt",inFile);
    FILE* out1 = fopen(buf1,"w");
  long idealSize = (long)(round((double)G->numVertices / (double)numColors));
                          
  for(long i=0; i < numColors; i++) {
 	fprintf(out1,"%ld \t %ld\n", idealSize, colorFreq[i]);
  }
  printf("Data written to file: %s\n", buf1);
   */
  
  printf("\n\n Balanced coloring (nT= %d):\n", nT);
  equitableDistanceOneColorBased(G, colors, numColors, colorFreq, nT, &tmpTime, 1);
    
#pragma omp parallel for
    for(long i = 0; i < numColors; i++) {
        colorFreq[i] = 0;
    }
    buildColorSize(NVer, colors, numColors, colorFreq);
    computeVariance(NVer, numColors, colorFreq);

    GiniCoeff = computeGiniCoefficient(colorFreq, numColors);
    printf("************************************************\n");
    printf("Gini coefficient: %g\n", GiniCoeff);
    printf("************************************************\n");
    
    
  /* Step 5: Clean up */
  free(G->edgeListPtrs);
  free(G->edgeList);
  free(G);

  //free(colorFreq);
  free(colors);

  return 0;

}
