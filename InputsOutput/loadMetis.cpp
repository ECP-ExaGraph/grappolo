
#include "input_output.h"
#include "defs.h"
#include "sstream"
#include "utilityStringTokenizer.hpp"


/**
 Metis format is the perfect format for what we used to have.
 It is dimacs10 format: where row number is vertex ID, and
 starting at 1.
 
 row# NumVertex NumEdge WeightIndication
 1	N1 W1 N2 W2 N3 W3...
 2	N1 W1 N2 W2 N3 W3...
 ...
 
 **/
void loadMetisFileFormat(graph *G, const char* filename) {
    long i, j, value, neighbor,  mNVer=0,  mNEdge=0;
    double edgeWeight, vertexWeight;
    std::string oneLine, myDelimiter(" "); //Delimiter is a blank space
    ifstream fin;
    char comment;
    
    fin.open(filename);
    if(!fin) {
        cerr<<"Within Function: loadMetisFileFormat() \n";
        cerr<<"Could not open the file.. \n";
        exit(1);
    }
    do { //Ignore the comment lines
        getline(fin, oneLine);
        comment = oneLine[0];
    } while ( comment == '%');
    
    StringTokenizer* ST = new StringTokenizer(oneLine, myDelimiter);
    if ( ST->HasMoreTokens() )
        mNVer  = atol( ST->GetNextToken().c_str() ); //Number of Vertices
    if ( ST->HasMoreTokens() )
        mNEdge  = atol( ST->GetNextToken().c_str() ); //Number of Edges
    if ( ST->HasMoreTokens() )
        value = atol( ST->GetNextToken().c_str() ); //Indication of the weights
    else
        value = 0;
    delete ST;
    //#ifdef PRINT_CF_DEBUG_INFO_
    cout<<"N Ver: "<<mNVer<<" N Edge: "<<mNEdge<<" value: "<<value<<" \n";
    //#endif
    
    long *mVerPtr   = (long *) malloc ((mNVer+1)  * sizeof(long)); //The Pointer
    edge *mEdgeList = (edge *) malloc ((2*mNEdge) * sizeof(edge)); //The Indices
    assert(mVerPtr != 0); assert(mEdgeList != 0);
    // printf("hi\n");
#pragma omp parallel for
    for (long i=0; i<=mNVer; i++) {
        mVerPtr[i] = 0;
    }
#pragma omp parallel for
    for (long i=0; i<(2*mNEdge); i++) {
        mEdgeList[i].tail   = -1;
        mEdgeList[i].weight = 0;
    }
    
    //Read the rest of the file:
    long PtrPos = 0, IndPos = 0, cumulative = 0;
    mVerPtr[PtrPos] = cumulative; PtrPos++;
    //printf("hi %d\n",value);
    switch ( value ) {
        case 0:  //No Weights
            //#ifdef PRINT_CF_DEBUG_INFO_
            cout<<"Graph Type: No Weights. \n";
            //#endif
            
            for ( i=0; i < mNVer; i++) {
                if ( fin.eof() )	    {
                    cout<<" Error reading the Metis input File \n";
                    cout<<" Reached Abrupt End \n";
                    exit(1);
                }
                j=0;
                std::getline (fin, oneLine);
                
                const auto strBegin = 0;
                const auto strEnd = oneLine.find_last_not_of(" \t");
                const auto strRang = strEnd-strBegin+1;
                if(strRang == 0)
                {
                    mVerPtr[i+1] = cumulative;
                    continue;
                }
                
                //	std::cout<< strBegin <<','<< strEnd <<','<< strRang<<std::endl;
                oneLine.resize(strRang);
                std::istringstream NewIss(oneLine);
                while (!NewIss.eof()) {
                    NewIss >> neighbor;
                    if(i == neighbor-1){
                        std::cout<< "self-edge removed"<<std::endl;
                        continue;
                    }
                    
                    j++;
                    
                    mEdgeList[IndPos].head = i;
                    mEdgeList[IndPos].tail = neighbor-1; //IndPos++;
                    mEdgeList[IndPos].weight = 1;
                    IndPos++;
                }
                cumulative += j;
                mVerPtr[i+1] = cumulative;  //Add to the Pointer Vector
            }
            cout<< "Total Edges:" <<cumulative<<endl;
            break;
            
        case 1: //Only Edge weights
            
            cout<<"Graph Type: Only Edge Weights. \n";
            for ( i=0; i < mNVer; i++) {
                if ( fin.eof() ){
                    cout<<" Error reading the Metis input File \n";
                    cout<<" Reached Abrupt End \n";
                    exit(1);
                }
                j=0;
                std::getline (fin, oneLine);
                
                const auto strBegin = 0;
                const auto strEnd = oneLine.find_last_not_of(" \t");
                const auto strRang = strEnd-strBegin+1;
                /*if(i == 59384)
                 printf("STRRANG %d\n", strRang);*/
                if(strRang <= 1)
                {
                    //		printf("%d\n",i);
                    mVerPtr[i+1] = cumulative;
                    continue;
                }
                
                oneLine.resize(strRang);
                std::istringstream NewIss(oneLine);
                while (!NewIss.eof()) {
                    NewIss >> neighbor;
                    NewIss >> edgeWeight;
                    
                    if(i == neighbor-1){
                        std::cout<< "self-edge removed"<<std::endl;
                        char test;
                        printf("%d %d %d %lf\n", i, IndPos,neighbor,edgeWeight);
                        //scanf("%c",&test);
                        continue;
                    }
                    j++;
                    
                    mEdgeList[IndPos].head = i;
                    mEdgeList[IndPos].tail = neighbor-1; //IndPos++;
                    mEdgeList[IndPos].weight = edgeWeight;
                    IndPos++;
                }
                cumulative += j;
                mVerPtr[i+1] = cumulative;  //Add to the Pointer Vector
            }
            break;
            
            /*  case 10: //Only Vertex Weights
             #ifdef PRINT_CF_DEBUG_INFO_
             cout<<"Graph Type: Only Vertex Weights. \n";
             cout<<"Will ignore vertex weights.\n";
             #endif
             
             for ( i=0; i < mNVer; i++)      {
             if ( fin.eof() )	{
             cout<<" Error reading the Metis input File \n";
             cout<<" Reached Abrupt End \n";
             exit(1);
             }
             j=0;
             getline (fin, oneLine);
             StringTokenizer* ST = new StringTokenizer(oneLine, myDelimiter);
             vertexWeight = atof( ST->GetNextToken().c_str() );
             while( ST->HasMoreTokens() )	{
             neighbor = (atol( ST->GetNextToken().c_str() ) - 1); //Zero-based index
             #ifdef PRINT_CF_DEBUG_INFO_
             cout<<"Neighbors: "<<neighbor<<" \t";
             #endif
             j++;
             mEdgeList[IndPos].tail = neighbor; IndPos++;
             mEdgeList[IndPos].weight = 1;
             }
             delete ST; //Clear the buffer
             cumulative += j;
             mVerPtr[PtrPos] = cumulative; PtrPos++; //Add to the Pointer Vector	      
             }
             break;
             case 11: //Both Edge and Vertex Weights:
             #ifdef PRINT_CF_DEBUG_INFO_
             cout<<"Graph Type: Both Edge and Vertex Weights. \n";
             #endif
             cout<<"Will ignore vertex weights.\n";
             
             for ( i=0; i < mNVer; i++) 		{
             if ( fin.eof() ) 	{
             cerr<<"Within Function: loadMetisFileFormat() \n";
             cerr<<" Error reading the Metis input File \n";
             cerr<<" Reached Abrupt End \n";
             exit(1);
             }
             j=0;
             getline (fin, oneLine);
             StringTokenizer* ST = new StringTokenizer(oneLine, myDelimiter);
             vertexWeight = atof( ST->GetNextToken().c_str() ); //Vertex weight
             while( ST->HasMoreTokens() )			{
             neighbor = (atol( ST->GetNextToken().c_str() )- 1); //Zero-based index
             mEdgeList[IndPos].tail = neighbor; IndPos++;
             edgeWeight = atof( ST->GetNextToken().c_str() );
             mEdgeList[IndPos].weight = (long)edgeWeight;  //Type casting
             j++;				
             }
             delete ST; //Clear the buffer
             cumulative += j;
             mVerPtr[PtrPos] = cumulative; PtrPos++; //Add to the Pointer Vector		
             }
             break;*/
    } //End of switch(value)
    fin.close();
    G->numVertices  = mNVer;
    G->sVertices    = mNVer;
    G->numEdges     = mNEdge;  //This is what the code expects
    G->edgeListPtrs = mVerPtr;  //Vertex Pointer
    G->edgeList     = mEdgeList;
    
} //End of loadMetisFileFormat()


