// construct a mesh from .csv files
template <unsigned int M, unsigned int N>
Mesh<M,N>::Mesh(const std::string& pointsFile,    const std::string& edgesFile,
		const std::string& trianglesFile, const std::string& neighborsFile){
  // open and parse CSV files
  CSVReader reader;
  CSVFile<double> points = reader.parseFile<double>(pointsFile);
  CSVFile<int> edges     = reader.parseFile<int>(edgesFile);
  CSVFile<int> triangles = reader.parseFile<int>(trianglesFile);
  CSVFile<int> neighbors = reader.parseFile<int>(neighborsFile);
    
  // set mesh internal representation to eigen matrix
  points_    = points.toEigen();
    
  // compute mesh limits
  for(size_t dim = 0; dim < N; ++dim){
    meshRange[dim].first  = points_.col(dim).minCoeff();
    meshRange[dim].second = points_.col(dim).maxCoeff();

    minMeshRange[dim] = meshRange[dim].first;
    kMeshRange[dim] = 1/(meshRange[dim].second - meshRange[dim].first);
  }
    
  // need to subtract 1 from all indexes since triangle indexes start from 1
  // C++ start counting from 0 instead
  edges_          = edges.toEigen();
  edges_          = (edges_.array() - 1).matrix();

  triangles_      = triangles.toEigen();
  triangles_      = (triangles_.array() -1).matrix();
  numElements     = triangles_.rows();
    
  // a negative value means no neighbor
  neighbors_      = neighbors.toEigen();
  neighbors_      = (neighbors_.array() - 1).matrix();
}

// build and provides a nice abstraction for an element given its ID
template <unsigned int M, unsigned int N>
const std::shared_ptr<Element<M,N>> Mesh<M,N>::requestElementById(unsigned int ID) const {
  // in the following use auto to take advantage of eigen acceleration
  
  // get the indexes of vertices from triangles_
  auto pointIndexes = triangles_.row(ID);
  // get neighbors information
  auto elementNeighbors = neighbors_.row(ID);
  
  // get vertices coordinates
  std::array<SVector<N>, N_VERTICES(M,N)> coords;
  std::array<unsigned int, M+1> neighbors;

  for(size_t i = 0; i < pointIndexes.size(); ++i){
    SVector<N> vertex(points_.row(pointIndexes[i]));
    coords[i] = vertex;

    // from triangle documentation: The first neighbor of triangle i is opposite the first corner of triangle i, and so on.
    // by storing neighboring informations as they come from triangle we have that neighbor[0] is the
    // triangle adjacent to the face opposite to coords[0]

    neighbors[i] = elementNeighbors[i];
  }
         
  return std::make_shared<Element<M,N>>(ID, coords, neighbors);
}
