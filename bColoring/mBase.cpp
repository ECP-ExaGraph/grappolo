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
ColorElem mBaseRedistribution(const Graph &g, ColorVector &colors, std::string input, ColorElem ncolors, int type)
{
	double t1,t2;
	t1 = mytimer();
	const GraphElem nv = g.getNumVertices();
	RandVec randVec(nv);
	generateRandomNumbers(randVec);

	srand(time(NULL));
	std::string outb;
	std::vector<std::string> outPutName(2);
	outPutName[0] = input+".LowerBoundedFF";
	outPutName[1] = input+".LowerBoundedLU";
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
	ColorVector done(ncolors,0);
	BitVector overSize(ncolors,false);
	GraphElem avg = ceil(nv/ncolors);
  
  if(avg> 1024+64)
    avg = 1024;


	// calculate the frequency 
	computeBinSizes(freq,baseColors,nv,ncolors);
	
	// Find the overSize bucket (can do some Optimization here)
	#pragma omp parallel for
	for(size_t ci = 0U; ci <ncolors; ci++)
		if(freq[ci]>(1088))
			overSize[ci]= true;

	/* Begining of Redistribution */
	std::cout << "VR start "<< std::endl;


	// Coloring Main Loop
	do {
                size_t qsz = q.size();
                qtmp.resize(qsz);
                double mst=0, cst=0;
                #pragma omp parallel default(none), shared(ncolors,freq,q, qtmp, qtmpPos, randVec, g, colors, qsz, nloops, nconflicts, std::cerr, type,baseColors,avg,overSize)//, MaxDegree)
                {
			// Travel unprocessed overfilled vertices
			#pragma omp for firstprivate(nloops, qsz), schedule(guided)
      for (GraphElem qi = 0L; qi < qsz; qi++) {
           GraphElem v = q[qi];
           ColorElem myColor=-1;
        	 bool threadDone = false;

          if(colors[v]!=-1){
            threadDone=false;
            for(myColor =0; myColor < ncolors; myColor++)
            {
              if(freq[myColor] < 1024)
                threadDone=true;
            }
          }
          

				if( (colors[v]==-1 || (freq[colors[v]]>avg && threadDone)) && overSize[baseColors[v]] == true){
                        	        ColorElem maxColor = -1;
                        	        BitVector mark(ncolors, false);
					// Mark the used color
					maxColor = distanceOneMarkArray(mark,g,v,colors);

					ColorElem permissable=0;
			
					//Pick the target
					if(type==0){ //first fit
            for(myColor =0; myColor < ncolors; myColor++)
            {
							  if(mark[myColor] != true && freq[myColor]<avg && overSize[myColor]!= true)
							    break;
             
            }
					}
					else if(type==1){ // Least Used
            myColor=-1;
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
//						#pragma omp atomic update
						freq[myColor]++;
						if(colors[v] != -1){
//							#pragma omp atomic update
							freq[colors[v]]--;
						}
						colors[v] = myColor;
					}
          


//          if(threadDone)
//            break;
				}	
			}// End of Vertex wise coloring (for)
		
			//Conflicts resloution step
			#pragma omp for firstprivate(qsz), schedule(guided)
			for (GraphElem qi = 0L; qi < qsz; qi++) {
				GraphElem v = q[qi];
				distanceOneConfResolution(g,v,colors,freq,randVec,qtmp,qtmpPos,1);
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
	std::cout << outb << " Least Time Time: " << t2-t1<<std::endl;

	//Sanity check;
//	distanceOneChecked(g,nv,colors);

	// Out put
//	outPut(colors,outb,freq,ncolors);	
}

