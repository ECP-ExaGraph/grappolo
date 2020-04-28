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

#include <stdio.h>
#include <stdlib.h>

using namespace std;

int main(int argc, char** argv) {
    
    //const int D1 = 6, D2 = 12;
    const int D1 = 8, D2 = 9;
    //const int D1 = 4, D2 = 4, D3 = 5;
    int D3 = 5;
    
    ////2-D Mesh:
    int self = -1, neighbor = -1;
    for (int row=0; row < D1; row++) {
        for (int col=0; col < D2; col++) {
            self = (row*D2) + col + 1; //Current node:
            ///////Add the four neighbors:
            //Left
            if(col == 0) { //First node in the column
                //Do nothing
            } else {
                neighbor = self - 1; //The left neighbor
                printf("%d %d\n", self, neighbor);
            }
            //Right
            if(col == (D2-1)) { //Last node in the column
                //Do nothing
            } else {
                neighbor = self + 1; //The left neighbor
                printf("%d %d\n", self, neighbor);
            }
            //Top
            if(row == 0) { //The first row of the grid
                //Do nothing
            } else {
                neighbor = self - D2; //The top neighbor
                printf("%d %d\n", self, neighbor);
            }
            //Bottom
            if(row == (D1-1)) { //The last row of the grid
                //Do nothing
            } else {
                neighbor = self + D2; //The bottom neighbor
                printf("%d %d\n", self, neighbor);
            }
        }
    }
    return 0;
    
    ////2-D Torus:
    self = -1;
    neighbor = -1;
    for (int row=0; row < D1; row++) {
        for (int col=0; col < D2; col++) {
            self = (row*D2) + col + 1; //Current node:
            ///////Add the four neighbors:
            //Left
            if(col == 0) { //First node in the column
                neighbor = (row*D2) + D2; //The last node on the row
            } else {
                    neighbor = self - 1; //The left neighbor
            }
            printf("%d %d\n", self, neighbor);
            //Right
            if(col == (D2-1)) { //Last node in the column
                neighbor = (row*D2) + 1; //The first node on the row
            } else {
                    neighbor = self + 1; //The left neighbor
            }
            printf("%d %d\n", self, neighbor);
            //Top
            if(row == 0) { //The first row of the grid
                neighbor = ((D1-1)*D2) + col + 1; //The last row neighbor of the grid
            } else {
                neighbor = self - D2; //The top neighbor
            }
            printf("%d %d\n", self, neighbor);
            
            //Bottom
            if(row == (D1-1)) { //The last row of the grid
                neighbor = col + 1; //The first row neighbot of the grid
            } else {
                neighbor = self + D2; //The bottom neighbor
            }
            printf("%d %d\n", self, neighbor);
            }
        }
    return 0;
    
    ////3-D Torus:
    self = -1; neighbor = -1;
    for (int depth = 0; depth < D3; depth++) { //Depth in the third dimension
        for (int row=0; row < D1; row++) { //Row
            for (int col=0; col < D2; col++) { //Column
                self = (D1*D2)*depth + (row*D2) + col + 1; //Current node
                ///////Add the six neighbors:
                ////FRONT
                if(depth == 0) { //First panel of the cube
                    neighbor = self + (D1*D2)*(D3-1); //The last panel of the cube
                } else {
                    neighbor = self - (D1*D2); //The corresponding node on the panel ahead
                }
                printf("%d %d\n", self, neighbor);
                
                ////BACK
                if(depth == (D3-1)) { //Last panel of the cube
                    neighbor = self - (D1*D2)*(D3-1); //The first panel of the cube
                } else {
                    neighbor = self + (D1*D2); //The corresponding node on the panel behind
                }
                printf("%d %d\n", self, neighbor);
                
                ////LEFT
                if(col == 0) { //First node in the column
                    neighbor = self + D2 - 1; //The last node on the row
                } else {
                    neighbor = self - 1; //The left neighbor
                }
                printf("%d %d\n", self, neighbor);
                
                ////RIGHT
                if(col == (D2-1)) { //Last node in the column
                    neighbor = self - col + 1; //The first node on the row
                } else {
                    neighbor = self + 1; //The left neighbor
                }
                printf("%d %d\n", self, neighbor);
                
                ////ABOVE
                if(row == 0) { //The first row of the panel
                    neighbor = self + ((D1-1)*D2); //The last row neighbor of the panel
                } else {
                    neighbor = self - D2; //The top neighbor
                }
                printf("%d %d\n", self, neighbor);
                
                ////BELOW
                if(row == (D1-1)) { //The last row of the grid
                    neighbor = self - ((D1-1)*D2); //The first row neighbor of the panel
                } else {
                    neighbor = self + D2; //The bottom neighbor
                }
                printf("%d %d\n", self, neighbor);
            }
        }
    }
    return 0;
}//End of main()
