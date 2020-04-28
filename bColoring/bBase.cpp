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
ColorElem schRedistribution(const Graph &g, ColorVector &colors, std::string input, ColorElem ncolors)
{
	double t1,t2;
	t1 = mytimer();
	const GraphElem nv = g.getNumVertices();
	RandVec randVec(nv);
	generateRandomNumbers(randVec);

	srand(time(NULL));
	std::string outb;
	outb = input+".SeR";

	// Rebuild indirection for coloring
	ColorVector colorPtr, colorIndex, freq;
	colorPtr.resize(ncolors+1);
	colorIndex.resize(nv);
	freq.resize(ncolors);
	buildColorsIndex(colors,ncolors,nv,colorPtr,colorIndex,freq);
	GraphElem avg = ceil(nv/ncolors);

	
	// MultiTries variable
	int shift = 0;
	int nmoved = 0;	
	// Conflicts check statistics
	ColorElem nconflicts=0;
	int nloops = 0;
	GraphElem realMaxDegree = -1;


	/* Begining of Redistribution */
	std::cout << "BatchMove start "<< std::endl;


	// Coloring Main Loop
	do{
		//Build the move queue
		MoveQueue marray;
		ColorVector newFreq=freq;
		for(ColorElem ci = 0; ci<ncolors;ci++){
			GraphElem stPost = 0;
			ColorElem counter = 0;
			if(newFreq[ci] > avg){	//Visit all other colors if overSize
				for(ColorElem ti= (ncolors-1-shift)%ncolors; counter<ncolors; ti--,counter++){
					if(newFreq[ci]<=avg)
						break;
					if( (ci!=ti) && newFreq[ti]<avg){
						const GraphElem numLeft = newFreq[ci]-avg;
						const GraphElem numToMove = std::min( (avg-newFreq[ti]),numLeft);
						MoveInfo m;
						m.source=ci;
						m.target=ti;
						m.numVertices = numToMove;
						m.startPost = stPost;
						stPost += numToMove;
						newFreq[ci] -= numToMove;
						newFreq[ti] += numToMove;
						marray.push_back(m);
					}
				}
			}
		}
		// Move the vertices in parallel
		const size_t mSize = marray.size();
		#pragma omp parallel default(none), shared(g,marray,colors,colorIndex,colorPtr), reduction(+:nmoved)
		{
			for(size_t mi = 0; mi <mSize; mi++){
				const MoveInfo& m = marray[mi];
				const GraphElem coloradj1 = colorPtr[m.source];
				const GraphElem coloradj2 = colorPtr[m.source + 1];
				
				GraphElem pstart = coloradj1 + m.startPost;
				GraphElem pend = std::min( (coloradj1+m.startPost+m.numVertices), coloradj2);

				#pragma omp for schedule(guided)
				for( GraphElem gi = pstart ;gi<pend; gi++){
					bool confl = false;
					GraphElem v = colorIndex[gi];
					GraphElem e0,e1;
					g.getEdgeRangeForVertex(v,e0,e1);
					while(e0<e1){
						const Edge &edge = g.getEdge(e0);
						if(colors[edge.tail] == m.target){
							confl = true;
							break;
						} 
						e0++;
					}
					if(!confl){
						colors[v] = m.target;
						nmoved++;
					}
				}

			}	
		}
		colorPtr.assign(ncolors+1,0);
		colorIndex.assign(nv,0);
		freq.assign(ncolors,0);
		buildColorsIndex(colors,ncolors,nv,colorPtr,colorIndex,freq);
		shift++;
	}while(false);
		
	t2 = mytimer();
	std::cout << outb << " Recoloring Time: " << t2-t1<<std::endl;

	//Sanity check;
	distanceOneChecked(g,nv,colors);

	// Out put
	
	std::cout<<"Total Number of Colors used: " << ncolors<<std::endl;
/*	for(int ci = 0; ci <nv; ci++)
	{
		std::cout<< ci << " : " <<colors[ci]<<std::endl;
	}*/
  
  ColorVector freq2(ncolors,0);
 
	computeBinSizes(freq2,colors,nv,ncolors);
  outPut(colors,outb,freq2,ncolors);	


}

