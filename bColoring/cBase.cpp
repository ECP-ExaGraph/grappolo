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

#include "coloringUtils.h"
//extern GraphElem MaxDegree;
/* The redistritbuted coloring step, no balance */
ColorElem cBaseRedistribution(graph* G, int* vtxColor, int ncolors, int type)
{
	printf("Color Base Redistribution")
	
	double time1=0, time2=0, totalTime=0;
	long NVer    = G->numVertices;
  long NEdge   = G->numEdges;  
  long *verPtr = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
	
	
	// Rebuild indirection for coloring
	ColorVector colorPtr, colorIndex, freq;
	colorPtr.resize(ncolors+1);
	colorIndex.resize(NVer);
	freq.resize(ncolors);
	
	buildColorsIndex(vtxColor, ncolors, NVer, colorPtr, colorIndex, freq);
	
	BitVector overSize(ncolors,false);
	long avg = ceil(nv/ncolors);
       
	// Find the overSize bucket (can do some Optimization here)
	#pragma omp parallel for
	for(int ci = 0U; ci <ncolors; ci++){
		if(freq[ci]>avg)
			overSize[ci]= true;
	}

	
	/* Begining of Redistribution */
	std::cout <<"AVG:"<<avg<< " CR start "<< std::endl;

		
	// Color Base Redist. 
	#pragma omp parallel default(none), shared(overSize, colorPtr,colorIndex,type,freq,avg,ncolors,g,colors,std::cerr,std::cout)
	{
		// Travel all colors
		for(ColorElem CI = 0; CI<ncolors && overSize[CI] ==true ;CI++){
			long cadj1	= colorPtr[CI];
			long cadj2 = colorPtr[CI+1];
			
			// Move the vetex in same bin together
			#pragma omp for schedule(guided)
			for(long ki=cadj1; ki<cadj2; ki++){
				long v = colorIndex[ki];
				
				if(freq[CI] <= avg)
					continue;
				
				//std::cout<<"Move: " <<v<<std::endl;
				// Build mark array for movement canadi.
				BitVector mark(ncolors,false);
				distanceOneMarkArray(mark,g,v,colors);
				if(colors[v]!=-1)
					mark[colors[v]] = true;	
					
				//Pick target
				ColorElem myColor = -1;
				if(type == 0){	//First Fit
					for(myColor = 0; myColor <ncolors; myColor++)
						if(mark[myColor] != true && freq[myColor]<avg)
							break;
				}else if(type == 1){ //Least Used
					for(ColorElem ci = 0; ci<ncolors; ci++)
						if(mark[ci]!=true && freq[ci] <avg)
							if(myColor == -1 || freq[myColor]>freq[ci])
								myColor = ci;
				}
				
				//Update the color
				if(myColor != -1 && myColor < ncolors){
					#pragma omp atomic update
					freq[myColor]++;
					#pragma omp atomic update
					freq[colors[v]]--;
					colors[v] = myColor;
				}
			}// End of singl color
		}//End of all colors.
	} //End of parallel step
	t2 = mytimer();
	std::cout << outb << " CRE Time: " << t2-t1<<std::endl;

	//Sanity check;
	distanceOneChecked(g,nv,colors);

	// Out put
	std::cout<<"Total Number of Colors used: " << ncolors<<std::endl;
	for(int ci = 0; ci <nv; ci++)
	{
		std::cout<< ci << " : " <<colors[ci]<<std::endl;
	}
	baseColors = colors;
}

