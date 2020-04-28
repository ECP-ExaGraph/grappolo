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
/* The basic coloring step, no balance */
ColorElem initColoringLU(const Graph &g, ColorVector &colors, std::string input)
{
  std::cout << "WELCOME TO THE LEAST USED ABI SHCHEM" <<std::endl;
	double t1,t2;
	t1 = mytimer();

	const GraphElem nv = g.getNumVertices();
	RandVec randVec(nv);
	generateRandomNumbers(randVec);

	srand(time(NULL));
	std::string outb;
	std::vector<std::string> outPutName(3);
	outPutName[0] =  input+".DefaultD";
	outPutName[1] =  input+".ALeastUsedD";
	outPutName[2] =  input+".ARandomD";
	
	// initialize the color to -1
	colors.resize(nv);
	#pragma omp parallel for default(none), shared(colors), schedule(static)
	for (GraphElem i = 0L; i < nv; i++)
		colors[i] = -1;
	
	// Set up all vertices in the queue
	ColorQueue q(nv), qtmp;
	GraphElem qtmpPos = 0L;
	#pragma omp parallel for default(none), shared(q), schedule(static)
	for (GraphElem i = 0; i < nv; i++)
		q[i] = i;
	ColorElem nconflicts = 0;
	int nloops = 0;
	GraphElem realMaxDegree = -1L;

	// Cal real maximum degree, not sure where it is used
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

//	std::cout << "Maximum degree for the graph: " << realMaxDegree << std::endl;

	// Because we do not know how much does the color really is
	ColorVector freq(MaxDegree,0);
	
	do {
		size_t qsz = q.size();
		qtmp.resize(qsz);
		double mst=0, cst=0;
		#pragma omp parallel default(none), shared(freq,q, qtmp, qtmpPos, randVec, g, colors, qsz, nloops, nconflicts, std::cerr)
		{
			// Coloring step
			#pragma omp for firstprivate(nloops, qsz), schedule(guided)
			for (GraphElem qi = 0L; qi < qsz; qi++) {
				GraphElem v = q[qi];
				ColorElem maxColor = -1;
				BitVector mark(MaxDegree, false);
				
				// Mark the array
				maxColor =  distanceOneMarkArray(mark, g, v, colors);
				ColorElem myColor = -1;
				ColorElem target ;
				ColorElem permissable = 0;
				
				// Pick the target
				//	Assefaw  least used
				for(target = 0; target<MaxDegree && freq[target]!=0; target++)
				{
					if(mark[target] != true)
						if(myColor == -1 || freq[myColor] > freq[target])
							myColor = target;
				}
				if(myColor == -1)
						myColor = target;
		
        #pragma omp atomic update
				freq[myColor]++;
				colors[v] = myColor;
			}// End of coloring	
		
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

    std::cout<<"Loop:" <<nloops<<std::endl;
		nconflicts += qtmpPos;
		nloops++;
		qtmpPos = 0;
	}while(!q.empty()); // End of the Coloring main loop
	t2 = mytimer();
//	std::cout << outb << " Initial Coloring Time: " << t2-t1<<std::endl;

	//Sanity check;
	distanceOneChecked(g,nv,colors);

	// Out put
	ColorElem ncolors = -1;
        size_t csz = colors.size();
        double variance;
        #pragma omp parallel for default(none), shared(colors), reduction(max: ncolors), firstprivate(csz), schedule(static)
        for (size_t ci = 0U; ci < csz; ci++)
        {
                if (colors[ci] > ncolors)
                        ncolors = colors[ci];
        }
	std::cout<<"Total Number of Colors used: " << ncolors<<std::endl;

	ncolors +=1;
	computeBinSizes(freq,colors,nv,ncolors);
//outPut(colors,outb,freq,ncolors);

  /*long dMax =0;
  for(int ci = 0; ci <nv; ci++)
  {
    int xP;
    xP = getDegree(ci,g);
    if(xP>dMax)
      dMax = xP;
  } 
	std::cout<< "Max Degree: " << dMax <<std::endl;*/
  return ncolors;
}

