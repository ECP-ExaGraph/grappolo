#include "input_output.h"

#include "H5Cpp.h"

#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

extern "C" {
#include "blosc_filter.h"
}


typedef long int GraphElem;
typedef std::pair<GraphElem, GraphElem> EdgePair;

void parse_EdgeListCompressedHDF5(graph * G, char *fileName) {
  printf("Parsing a HDF5 file compressed using the BLOSC-C compression library...\n");
  printf("WARNING: Assumes that the graph is directed -- an edge is stored only once.\n");
  printf("       : Graph will be stored as undirected, each edge appears twice.\n");

  int nthreads = 0;
  double time1, time2;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  printf("parse_EdgeListCompressedHDF5: Number of threads: %d\n ", nthreads);

  char *version, *date;
  /* Register the filter with the library */
  int r = register_blosc(&version, &date);
  //printf("Blosc version info: %s (%s)\n", version, date);

  std::string fname(fileName);
  const H5std_string h5fn(fname); //fname should be graphname.h5
  std::string bfname = fname.substr(fname.find_last_of("/\\") + 1);
  bfname = bfname.substr(0, bfname.find_last_of(".")); //bfname should be graphname
  const H5std_string DATASET_NAME(bfname);

  time1 = omp_get_wtime();

  int numfilt;
  size_t nelmts = {1}, namelen = {1};
  unsigned flags, filter_info, cd_values[1], idx;
  char name[1];
  H5Z_filter_t filter_type;

  // Open the file and the dataset in the file.
  H5File file(h5fn, H5F_ACC_RDONLY);
  DataSet *dataset = new DataSet(file.openDataSet(DATASET_NAME));

  // Get the create property list of the dataset.
  DSetCreatPropList *plist = new DSetCreatPropList(dataset->getCreatePlist());

  // Get the number of filters associated with the dataset.
  numfilt = plist->getNfilters();
  std::cout << "Number of filters associated with dataset: " << numfilt << std::endl;

  for (idx = 0; idx < numfilt; idx++) {
    nelmts = 0;

    filter_type = plist->getFilter(idx, flags, nelmts, cd_values, namelen, name,
                                   filter_info);

    std::cout << "Filter Type: ";

    switch (filter_type) {
      case H5Z_FILTER_DEFLATE:
        std::cout << "H5Z_FILTER_ZLIB_DEFLATE" << std::endl;
        break;
      case H5Z_FILTER_SZIP:
        std::cout << "H5Z_FILTER_SZIP" << std::endl;
        break;
      default:
        std::cout << "Other filter type (BLOSC) included." << std::endl;
    }
  }

  /*
   * Get dataspace of the dataset.
   */
  DataSpace dataspace = dataset->getSpace();

  /*
   * Get the dimension size of each dimension in the dataspace and
   * display them.
   */
  hsize_t dims_out[1];
  const int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);

  const GraphElem NE = dims_out[0] / 2 - 1;

  // Read data.
  std::vector <EdgePair> edgeListH5(NE + 1);
  dataset->read(edgeListH5.data(), PredType::NATIVE_LONG);

  time2 = omp_get_wtime();

  std::cout << "HDF5 File parsing time: " << (time2 - time1) << std::endl;

  delete plist;
  delete dataset;
  file.close();

  GraphElem NV = (edgeListH5[0]).first;
  assert(NE == (edgeListH5[0]).second);

  std::cout << "Loaded dataset: " << fname << "\n ";
  printf("|V|= %ld, |E|= %ld \n", NV, NE);
  printf("Weights are set to positive one.\n");

  time1 = omp_get_wtime();
  long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
  assert(edgeListPtr != NULL);
  edge *edgeList = (edge *) malloc( 2*NE * sizeof(edge)); //Every edge stored twice
  assert( edgeList != NULL);
  time2 = omp_get_wtime();
  printf("Time for allocating memory for storing graph = %lf\n", time2 - time1);
#pragma omp parallel for
  for (long i=0; i <= NV; i++)
    edgeListPtr[i] = 0; //For first touch purposes

  //////Build the EdgeListPtr Array: Cumulative addition
  time1 = omp_get_wtime();
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    __sync_fetch_and_add(&edgeListPtr[edgeListH5[i+1].first], 1); //Leave 0th position intact
    __sync_fetch_and_add(&edgeListPtr[edgeListH5[i+1].second], 1);
  }
  assert(edgeListPtr[0] == 0); //Make sure the vertex ids are 1-based numbers
  for (long i=0; i<NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = omp_get_wtime();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld\n", NE*2, edgeListPtr[NV]);
  printf("*********** (%ld)\n", NV);

  printf("About to build edgeList...\n");
  time1 = omp_get_wtime();
  //Keep track of how many edges have been added for a vertex:
  long  *added  = (long *)  malloc( NV  * sizeof(long)); assert( added != NULL);
#pragma omp parallel for
  for (long i = 0; i < NV; i++)
    added[i] = 0;

  //Build the edgeList from edgeListTmp:
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    long head      = edgeListH5[i+1].first - 1;
    long tail      = edgeListH5[i+1].second - 1;
    double weight  = 1.0;
    assert((head>=0)&&(tail>=0));

    long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);
    edgeList[Where].head = head;
    edgeList[Where].tail = tail;
    edgeList[Where].weight = weight;
    //Now add the counter-edge:
    Where = edgeListPtr[tail] + __sync_fetch_and_add(&added[tail], 1);
    edgeList[Where].head = tail;
    edgeList[Where].tail = head;
    edgeList[Where].weight = weight;
  }
  time2 = omp_get_wtime();
  printf("Time for building edgeList = %lf\n", time2 - time1);

  G->sVertices    = NV;
  G->numVertices  = NV;
  G->numEdges     = NE;
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;

  //Clean up
  edgeListH5.clear();
  free(added);

}//End of parse_Dimacs9FormatDirectedNewD()

void parse_EdgeListCompressedHDF5NoDuplicates(graph * G, char *fileName) {
    printf("Parsing a HDF5 file compressed using the BLOSC-C compression library...\n");
    printf("WARNING: Assumes that the graph is directed -- an edge is stored only once.\n");
    printf("       : Graph will be stored as undirected, each edge appears twice.\n");
    
    int nthreads = 0;
    double time1, time2;
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    printf("parse_EdgeListCompressedHDF5: Number of threads: %d\n ", nthreads);
    
    char *version, *date;
    /* Register the filter with the library */
    int r = register_blosc(&version, &date);
    //printf("Blosc version info: %s (%s)\n", version, date);
    
    std::string fname(fileName);
    const H5std_string h5fn(fname); //fname should be graphname.h5
    std::string bfname = fname.substr(fname.find_last_of("/\\") + 1);
    bfname = bfname.substr(0, bfname.find_last_of(".")); //bfname should be graphname
    const H5std_string DATASET_NAME(bfname);
    
    time1 = omp_get_wtime();
    
    int numfilt;
    size_t nelmts = {1}, namelen = {1};
    unsigned flags, filter_info, cd_values[1], idx;
    char name[1];
    H5Z_filter_t filter_type;
    
    // Open the file and the dataset in the file.
    H5File file(h5fn, H5F_ACC_RDONLY);
    DataSet *dataset = new DataSet(file.openDataSet(DATASET_NAME));
    
    // Get the create property list of the dataset.
    DSetCreatPropList *plist = new DSetCreatPropList(dataset->getCreatePlist());
    
    // Get the number of filters associated with the dataset.
    numfilt = plist->getNfilters();
    std::cout << "Number of filters associated with dataset: " << numfilt << std::endl;
    
    for (idx = 0; idx < numfilt; idx++) {
        nelmts = 0;
        
        filter_type = plist->getFilter(idx, flags, nelmts, cd_values, namelen, name,
                                       filter_info);
        
        std::cout << "Filter Type: ";
        
        switch (filter_type) {
            case H5Z_FILTER_DEFLATE:
                std::cout << "H5Z_FILTER_ZLIB_DEFLATE" << std::endl;
                break;
            case H5Z_FILTER_SZIP:
                std::cout << "H5Z_FILTER_SZIP" << std::endl;
                break;
            default:
                std::cout << "Other filter type (BLOSC) included." << std::endl;
        }
    }
    
    /*
     * Get dataspace of the dataset.
     */
    DataSpace dataspace = dataset->getSpace();
    
    /*
     * Get the dimension size of each dimension in the dataspace and
     * display them.
     */
    hsize_t dims_out[1];
    const int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
    
    const GraphElem NE = dims_out[0] / 2 - 1;
    
    // Read data.
    std::vector <EdgePair> edgeListH5(NE + 1);
    dataset->read(edgeListH5.data(), PredType::NATIVE_LONG);
    
    time2 = omp_get_wtime();
    
    std::cout << "HDF5 File parsing time: " << (time2 - time1) << std::endl;
    
    delete plist;
    delete dataset;
    file.close();
    
    GraphElem NV = (edgeListH5[0]).first;
    assert(NE == (edgeListH5[0]).second);
    
    std::cout << "Loaded dataset: " << fname << "\n ";
    printf("|V|= %ld, |E|= %ld \n", NV, NE);
    printf("Weights are set to positive one.\n");
    time1 = omp_get_wtime();
    edge *tmpEdgeList = (edge *) malloc( 2*NE * sizeof(edge)); //Every edge stored twice
#pragma omp parallel for
    for(long i=0; i<NE; i++) {
        tmpEdgeList[2*i].head = edgeListH5[i+1].first; //Leave 0th position intact
        tmpEdgeList[2*i].tail = edgeListH5[i+1].second;
        tmpEdgeList[2*i].weight = 1.0; //Set weight to one
        
        tmpEdgeList[2*i+1].head = edgeListH5[i+1].second; //Leave 0th position intact
        tmpEdgeList[2*i+1].tail = edgeListH5[i+1].first;
        tmpEdgeList[2*i+1].weight = 1.0; //Set weight to one
    }
    long newNE = removeEdges(NV, 2*NE, tmpEdgeList); //Each edge is stored twice
    edgeListH5.clear();
    time2 = omp_get_wtime();
    printf("Number of duplicates removed: %d\n", (2*NE - newNE)/2);
    printf("New number of edges: %d\n", newNE/2);
    printf("Time for removing duplicates = %lf\n", time2 - time1);
    
    time1 = omp_get_wtime();
    long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
    assert(edgeListPtr != NULL);
    edge *edgeList = (edge *) malloc( newNE * sizeof(edge)); //Every edge stored twice
    assert( edgeList != NULL);
    time2 = omp_get_wtime();
    printf("Time for allocating memory for storing graph = %lf\n", time2 - time1);
#pragma omp parallel for
    for (long i=0; i <= NV; i++)
    edgeListPtr[i] = 0; //For first touch purposes
    
    //////Build the EdgeListPtr Array: Cumulative addition
    time1 = omp_get_wtime();
#pragma omp parallel for
    for(long i=0; i<newNE; i++) {
        __sync_fetch_and_add(&edgeListPtr[tmpEdgeList[i].head], 1); //Nodes are 1-based indices
        //__sync_fetch_and_add(&edgeListPtr[tmpEdgeList[i].tail], 1); //Every edge stored twice
    }
    assert(edgeListPtr[0] == 0); //Make sure the vertex ids are 1-based numbers
    for (long i=0; i<NV; i++) {
        edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
    }
    //The last element of Cumulative will hold the total number of characters
    time2 = omp_get_wtime();
    printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
    printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld\n", newNE, edgeListPtr[NV]);
    printf("*********** (%ld)\n", NV);
    
    printf("About to build edgeList...\n");
    time1 = omp_get_wtime();
    //Keep track of how many edges have been added for a vertex:
    long  *added  = (long *)  malloc( NV  * sizeof(long)); assert( added != NULL);
#pragma omp parallel for
    for (long i = 0; i < NV; i++)
    added[i] = 0;
    
    //Build the edgeList from edgeListTmp:
#pragma omp parallel for
    for(long i=0; i<newNE; i++) {
        long head      = tmpEdgeList[i].head - 1;
        long tail      = tmpEdgeList[i].tail - 1;
        assert((head>=0)&&(tail>=0));
        assert((head<NV)&&(tail<NV));
        
        long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);
        edgeList[Where].head = head;
        edgeList[Where].tail = tail;
        edgeList[Where].weight = 1.0;
        //Now add the counter-edge:
        //Where = edgeListPtr[tail] + __sync_fetch_and_add(&added[tail], 1);
        //edgeList[Where].head = tail;
        //edgeList[Where].tail = head;
        //edgeList[Where].weight = weight;
    }
    time2 = omp_get_wtime();
    printf("Time for building edgeList = %lf\n", time2 - time1);
    
    G->sVertices    = NV;
    G->numVertices  = NV;
    G->numEdges     = newNE / 2;
    G->edgeListPtrs = edgeListPtr;
    G->edgeList     = edgeList;
    
    //Clean up
    free(tmpEdgeList);
    free(added);
    
}//End of parse_Dimacs9FormatDirectedNewD()
