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
ColorElem reColor(const Graph &g, ColorVector &baseColors, std::string input, ColorElem ncolors,double factor)
{
	double t1,t2;
	t1 = mytimer();
	const GraphElem nv = g.getNumVertices();
	RandVec randVec(nv);
	generateRandomNumbers(randVec);

	srand(time(NULL));
	std::string outb;
	outb = input+".ReC"+std::to_string(factor);

	// Rebuild indirection for coloring
	ColorVector colorPtr, colorIndex, freq;
	colorPtr.resize(ncolors+1);
	colorIndex.resize(nv);
	freq.resize(ncolors);
	buildColorsIndex(baseColors,ncolors,nv,colorPtr,colorIndex,freq);
	GraphElem avg = ceil(nv/ncolors);

	// initialize the color -1, prepare for recolor
	ColorVector newColors(nv);
	ColorVector newFreq(MaxDegree,0);
	#pragma omp parallel for default(none), shared(newColors,baseColors), schedule(static)
	for(GraphElem i = 0L; i<nv;i++)
		newColors[i]= -1;

	// reverse the vertex from highest color to lowest color
	ColorQueue q(nv), qtmp;
	GraphElem qtmpPos = 0L;
	#pragma omp parallel for default(none), shared(q,colorIndex),schedule(static)
	for (GraphElem i = 0; i < nv; i++)
		q[i] = colorIndex[nv-1-i];

	// Conflicts check statistics
	ColorElem nconflicts=0;
	int nloops = 0;
	GraphElem realMaxDegree = -1;

	/* Cal real Maximum degree, no used
	#pragma omp parallel for default(none), shared(g), reduction(max: realMaxDegree), schedule(static)
        for (GraphElem i = 0L; i < nv; i++) {
                GraphElem e0, e1, er;
                g.getEdgeRangeForVertex(i, e0, e1);
                er = e1 - e0;
                if (er > realMaxDegree)
                        realMaxDegree = er;
        }
        static_assert(sizeof(int) == sizeof(int32_t), "int should be 32-bit in size");
        assert((realMaxDegree < INT32_MAX) && (realMaxDegree > 0L));*/


	/* Begining of Redistribution */
	std::cout << "ReColor start "<< std::endl;


	// Coloring Main Loop
	do {
                size_t qsz = q.size();
                qtmp.resize(qsz);
                double mst=0, cst=0;
                #pragma omp parallel default(none), shared(ncolors,newFreq,q, qtmp, qtmpPos, randVec, g, newColors, qsz, nloops, nconflicts, std::cerr, avg,factor)//, MaxDegree)
                {
			// Travel unprocessed overfilled vertices
			#pragma omp for firstprivate(nloops, qsz), schedule(guided)
                        for (GraphElem qi = 0L; qi < qsz; qi++) {
                                GraphElem v = q[qi];
				ColorElem maxColor = -1;
                        	BitVector mark(MaxDegree, false);
				// Mark the used color
				maxColor = distanceOneMarkArray(mark,g,v,newColors);
				ColorElem myColor=-1;
				
				//Pick the target
				for(myColor =0; myColor <MaxDegree; myColor++){
					if(mark[myColor] != true && newFreq[myColor]<(avg)){
						break;
					}
				}	
			
				if(myColor == MaxDegree){
					std::cerr<< "Increase too much color, please check the code" << std::endl;
					exit(1);
				}
                                newColors[v] = myColor;
				#pragma omp atomic update
				newFreq[myColor] ++;
			}// End of Vertex wise coloring (for)
		
			//Conflicts resloution step
			#pragma omp for firstprivate(qsz), schedule(guided)
			for (GraphElem qi = 0L; qi < qsz; qi++) {
				GraphElem v = q[qi];
				distanceOneConfResolution(g,v,newColors,newFreq,randVec,qtmp,qtmpPos,1);
			} //End of identify all conflicts (re-set conflicts to -1)
			#pragma omp single
			{
				q.resize(qtmpPos);
			}
			#pragma omp for schedule(static)
			for (GraphElem qi = 0; qi<qtmpPos;qi++)
				q[qi] = qtmp[qi];
		} //End of parallel step

		nconflicts += qtmpPos;
		nloops++;
		qtmpPos = 0;
	}while(!q.empty()); // End of the Coloring main loop
	t2 = mytimer();
	std::cout << outb << " Recoloring Time: " << t2-t1<<std::endl;

	//Sanity check;
	distanceOneChecked(g,nv,newColors);

	// Out put
        ColorElem newNcolors = -1;
        double variance;
        #pragma omp parallel for default(none), shared(newColors), reduction(max: newNcolors), schedule(static)
        for (size_t ci = 0U; ci < nv; ci++)
        {
                if (newColors[ci] > newNcolors)
                        newNcolors = newColors[ci];
        }
	ColorVector newFreq2(MaxDegree,0);
  computeBinSizes(newFreq2,newColors,nv,newNcolors);

	outPut(newColors,outb,newFreq2,newNcolors);
	/*	
	std::cout<<"Total Number of Colors used: " << newNcolors<<std::endl;
	for(int ci = 0; ci <nv; ci++)
	{
		std::cout<< ci << " : " <<newColors[ci]<<std::endl;
	}*/
	baseColors = newColors;
}

