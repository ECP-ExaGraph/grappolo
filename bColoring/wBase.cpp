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
ColorElem wBaseRedistribution(const Graph &g, ColorVector &colors, std::string input, ColorElem ncolors, int type)
{
	double t1,t2;
	t1 = mytimer();
	const GraphElem nv = g.getNumVertices();
  const GraphElem ne = g.getNumEdges();
	RandVec randVec(nv);
	generateRandomNumbers(randVec);

	srand(time(NULL));
	std::string outb;
	std::vector<std::string> outPutName(2);
	outPutName[0] = input+".before";
	outPutName[1] = input+".WeighTed";
	outb = outPutName[type];

	// initialize the color to baseColor
	ColorVector baseColors(nv);
	#pragma omp parallel for default(none), shared(colors,baseColors), schedule(static)
	for(GraphElem i = 0L; i<nv;i++)
		baseColors[i] = colors[i];

	// Put uncolor vertices in the queue
	ColorQueue q(nv), qtmp;
	GraphElem qtmpPos = 0L;
	#pragma omp parallel for default(none), shared(q),schedule(static)
	for (GraphElem i = 0; i < nv; i++)
		q[i] = i;

	// Conflicts check statistics
	ColorElem nconflicts=0;
	int nloops = 0;
	GraphElem realMaxDegree = -1;

	// Cal real Maximum degree, no used
	#pragma omp parallel for default(none), shared(g), reduction(max: realMaxDegree), schedule(static)
        for (GraphElem i = 0L; i < nv; i++) {
                GraphElem e0, e1, er;
                g.getEdgeRangeForVertex(i, e0, e1);
                er = e1 - e0;
                if (er > realMaxDegree)
                        realMaxDegree = er;
        }
        static_assert(sizeof(int) == sizeof(int32_t), "int should be 32-bit in size");
        assert((realMaxDegree < INT32_MAX) && (realMaxDegree > 0L));

	// Holder for frequency, could use realMaxDegree here
	ColorVector freq(ncolors,0);
	ColorVector freq2(ncolors,0);
	BitVector overSize(ncolors,false);
	GraphElem avg = ceil(ne/ncolors);
        


  std::cout<<"AVG:"<<avg<<std::endl;

	// calculate the frequency 
	computeBinSizesWeighted(freq,baseColors,nv,ncolors,g);
	//outPut(colors,outPutName[0],freq,ncolors);	
	// Find the overSize bucket (can do some Optimization here)
	#pragma omp parallel for
	for(size_t ci = 0U; ci <ncolors; ci++){
  //  std::cout<<freq[ci]<<std::endl;
		if(freq[ci]>avg)
			overSize[ci]= true;
  }
	/* Begining of Redistribution */
	std::cout << "VR start "<< std::endl;


	// Coloring Main Loop
	do {
                size_t qsz = q.size();
                qtmp.resize(qsz);
                double mst=0, cst=0;
                #pragma omp parallel default(none), shared(ncolors,freq,q, qtmp, qtmpPos, randVec, g, colors, qsz, nloops, nconflicts, std::cerr, type,baseColors,avg,overSize, std::cout) //, MaxDegree)
                {
			// Travel unprocessed overfilled vertices
			#pragma omp for firstprivate(nloops, qsz), schedule(guided)
                        for (GraphElem qi = 0L; qi < qsz; qi++) {
                                GraphElem v = q[qi];
                                ColorElem vDeg = getDegree(v,g);
				
				
				if( (colors[v]==-1 || freq[colors[v]]>avg) && overSize[baseColors[v]] == true){
                        	        ColorElem maxColor = -1;
                        	        BitVector mark(ncolors, false);
					// Mark the used color
					maxColor = distanceOneMarkArray(mark,g,v,colors);

					ColorElem myColor=-1;
					ColorElem permissable=0;
			
					//Pick the target
					if(type==0){ //first fit
						for(myColor =0; myColor < ncolors; myColor++)
							if(mark[myColor] != true && freq[myColor]<avg && overSize[myColor]!= true)
              {
                //std::cout<< myColor <<"," <<colors[v]<<std::endl;
							  break;
              }
					}
					else if(type==1){ // Least Used
						for(ColorElem ci = 0; ci<ncolors;ci++){
							if(mark[ci] != true && freq[ci]<avg && overSize[ci]!=true){
								if(myColor==-1||freq[myColor]>freq[ci]){
									myColor = ci;
								}
							}
						}
					}
					// Go back to original color if no where to go after conf.
					if(colors[v]==-1 && (myColor==-1 || myColor ==ncolors) )
						myColor=baseColors[v];

					// Move to the new color if avaliable
					if(myColor != ncolors && myColor !=-1){
						#pragma omp atomic update 
						freq[myColor]+= vDeg;
						if(colors[v] != -1){
							#pragma omp atomic update
							freq[colors[v]]-= vDeg;
						}
						colors[v] = myColor;
					}
				}	
			}// End of Vertex wise coloring (for)
		
			//Conflicts resloution step
			#pragma omp for firstprivate(qsz), schedule(guided)
			for (GraphElem qi = 0L; qi < qsz; qi++) {
				GraphElem v = q[qi];
				distanceOneConfResolutionWeighted(g,v,colors,freq,randVec,qtmp,qtmpPos,1);
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
	std::cout << outb << "Weighted  Time: " << t2-t1<<std::endl;

	//Sanity check;
//	distanceOneChecked(g,nv,colors);

	// Out put
	//computeBinSizes(freq2,colors,nv,ncolors);
  //outPut(colors,outPutName[1],freq2,ncolors);	
}

