#ifndef __TREE_SEARCH_H__
#define __TREE_SEARCH_H__

#include <cstddef>
#include <list>
#include <stack>
#include <utility>

#include "Tree.h"
#include "../../../OPT/Utils.h"
#include "Query.h"
#include "Mesh.h"

// a specific node data structure for easy management of the ADT during element search
template <unsigned int N>
struct ADTnode {
  unsigned int elementID_;  // the element ID to which this node referes to
  SVector<N> point_;        // the point stored in this node 
  rectangle<N> range_;      // the range in the unit hypercube this node refers to
  
  // constructor
  ADTnode(unsigned int elementID, const SVector<N>& point, const rectangle<N>& range)
    : elementID_(elementID), point_(point), range_(range) {}
};

// Alternating Decision Tree implementation for tree-based search of elements over an unstructured mesh
template <unsigned int N, unsigned int M>
class ADTSearch{
private:
  Tree<ADTnode<2*N>> tree;

  Mesh<N,M>& mesh_;

  // build an Alternating Decision Tree given a set of N-dimensional points
  void init(const std::vector<std::pair<SVector<2*N>, unsigned int>>& data);
  
  // perform a geometric search returning all points which lie in a given query
  std::list<unsigned int> geometricSearch(const Query<2*N>& query);
  
public:
  // initialize the ADT using informations coming from the mesh
  ADTSearch(Mesh<N,M>& mesh) : mesh_(mesh){

    // move mesh elements to 2N dimensional points
    std::vector<std::pair<SVector<2*N>, unsigned int>> data;
    data.reserve(mesh_.getNumberOfElements()); // avoid useless reallocations at runtime
    
    for(std::shared_ptr<Element<N,M>> element : mesh_){
      // compute bounding box
      std::pair<SVector<N>, SVector<N>> boundingBox = element->computeBoundingBox();
    
      // create 2N dimensional point
      SVector<2*N> elementToPoint;
      
      // scale dimensions in the unit hypercube
      // point scaling means to apply the following linear transformation to each dimension of the point
      // scaledPoint[dim] = (point[dim] - meshRange[dim].first)/(meshRange[dim].second - meshRange[dim].first)
      // we can use cached results to speed up the computation and compute
      // scaledPoint[dim] = (point[dim] - minMeshRange[dim])*kMeshRange[dim]
      
      std::array<double, N> minMeshRange = mesh_.getMinMeshRange();
      std::array<double, N> kMeshRange   = mesh_.getKMeshRange();

      for(size_t dim = 0; dim < N; ++dim){
	boundingBox.first[dim]  = (boundingBox.first[dim]  - minMeshRange[dim])*kMeshRange[dim];
	boundingBox.second[dim] = (boundingBox.second[dim] - minMeshRange[dim])*kMeshRange[dim];
      }
      
      elementToPoint << boundingBox.first, boundingBox.second;
      data.push_back(std::make_pair(elementToPoint, element->getID()));
    }
    
    // set up internal data structure
    init(data);
  }
  
  // applies the ADT geometric search to return the mesh element containing a given point
  std::shared_ptr<Element<N, M>> search(const SVector<N>& point);
  
  // getters
  Tree<ADTnode<2*N>> getTree() const { return tree; }
};

// data needs not to be rescaled before calling this method. Rescaling of the points in the unit hypercube is handled
// here as part of the ADT construction
template <unsigned int N, unsigned int M>
void ADTSearch<N,M>::init(const std::vector<std::pair<SVector<2*N>, unsigned int>>& data) {
  
  // initialize here once
  SVector<2*N> left_lower_corner = SVector<2*N>::Zero(), right_upper_corner = SVector<2*N>::Ones();
  
  // initialize tree data structure
  tree = Tree(ADTnode<2*N>(data[0].second, data[0].first, std::make_pair(left_lower_corner, right_upper_corner)));
  
  // process all points inside data one by one and insert them in the correct position
  for(size_t j = 1; j < data.size(); ++j){
    SVector<2*N> nodeData    = data[j].first;
    unsigned int nodeID      = data[j].second;
    rectangle<2*N> nodeRange = std::make_pair(left_lower_corner, right_upper_corner);

    // traverse the tree based on coordinates of data and insert the corresponding node at right position in the tree
    node_ptr<ADTnode<2*N>> current = tree.getNode(0); // root node
    
    bool inserted = false;           // stop iterating when an insertion point has been found
    unsigned int iteration = 1;      // defines the granularity of the split. split points are located at (0.5)^iteration
    std::array<double,2*N> offset{}; // keep track of the splits of the domain at each iteration
  
    // search for the right insertion location in the tree
    while(!inserted){
      for(size_t dim = 0; dim < 2*N; ++dim){     // cycle over dimensions
	double split_point = offset[dim] + std::pow(0.5, iteration); // split point
	if(nodeData[dim] < split_point){
      	  nodeRange.second[dim] = split_point;   // shrink node range on the left
	  if(tree.insert(ADTnode<2*N>(nodeID, nodeData, nodeRange), current->getKey(), LinkDirection::LEFT )){ // O(1) operation
	    inserted = true;                     // stop searching for location
	    break;
	  }else
	    current = current->getChildren()[0]; // move to left child
	}
	else{
	  nodeRange.first[dim] = split_point;    // shrink node range on the right
	  if(tree.insert(ADTnode<2*N>(nodeID, nodeData, nodeRange), current->getKey(), LinkDirection::RIGHT)){ // O(1) operation
	    inserted = true;                     // stop searching for location
	    break;
	  }else{
	    current = current->getChildren()[1]; // move to right child
	    offset[dim] += std::pow(0.5, iteration);
	  }
	}
      }
      // virtually perform an half split of the hyper-cube
      iteration++;
    }
  }
  // construction ended
  return;
}

// a searching range (here called query) is supplied as a pair of points (a,b) where a is the
// lower-left corner and b the upper-right corner of the query rectangle. This method find all the points
// which are contained in a given query
template <unsigned int N, unsigned int M>
std::list<unsigned int> ADTSearch<N,M>::geometricSearch(const Query<2*N> &query) {
  // initialize result
  std::list<unsigned int> searchResult;
  
  // use a stack to assist the searching process
  std::stack< node_ptr<ADTnode<2*N>> > stack;
  stack.push(tree.getNode(0)); // start from root
  
  while(!stack.empty()){
    node_ptr<ADTnode<2*N>> current = stack.top();
    stack.pop();
        
    // add to solution if point is contained in query range
    if(query.contains(current->getData().point_))
      searchResult.push_back(current->getData().elementID_);
    
    // get children at node
    std::array<node_ptr<ADTnode<2*N>>, 2> children = current->getChildren();
    
    bool left_child_test  = children[0] != nullptr ? query.intersect(children[0]->getData().range_) : false;
    bool right_child_test = children[1] != nullptr ? query.intersect(children[1]->getData().range_) : false;
    
    if(left_child_test)        // test if left  child range intersects query range
      stack.push(children[0]); 
    if(right_child_test)       // test if right child range intersects query range
      stack.push(children[1]); 
  }
  
  // search completed
  return searchResult;
}

// once mesh elements are mapped as points in a 2N dimensional space, the problem of searching for the
// element containing a given point can be solved as a geometric search problem in a 2N dimensional space
template <unsigned int N, unsigned int M>
std::shared_ptr<Element<N, M>> ADTSearch<N,M>::search(const SVector<N> &point) {

  // map input point in the unit hypercube
  std::array<double, N> minMeshRange = mesh_.getMinMeshRange();
  std::array<double, N> kMeshRange   = mesh_.getKMeshRange();
  
  // point scaling means to apply the following linear transformation to each dimension of the point
  // scaledPoint[dim] = (point[dim] - meshRange[dim].first)/(meshRange[dim].second - meshRange[dim].first)
  // we can use cached results to speed up the computation and compute
  // scaledPoint[dim] = (point[dim] - minMeshRange[dim])*kMeshRange[dim]
  
  SVector<N> scaledPoint;
  for(size_t dim = 0; dim < N; ++dim){
    scaledPoint[dim] = (point[dim] - minMeshRange[dim])*kMeshRange[dim];
  }
  
  // build search query
  SVector<2*N> lower, upper;
  lower << SVector<N>::Zero(), scaledPoint;
  upper << scaledPoint, SVector<N>::Ones();
  rectangle<2*N> query = std::make_pair(lower,upper);
    
  // perform search (now the problem has been transformed to the one of searching for the set
  // of points contained in the range of the query. See "(J. Bonet, J. Peraire) 1991
  // An alternating digital tree (ADT) algorithm for 3D geometric searching and intersection problems"
  // for details)
  std::list<unsigned int> searchResult = geometricSearch(Query<2*N>(query));
  
  // exhaustively scan the query results to get the searched mesh element
  for(unsigned int ID : searchResult){
    std::shared_ptr<Element<N,M>> element = mesh_.requestElementById(ID);
    if(element->contains(point)){
      return element;
    }
  }
  
  // no element found (this will rise an Address bounday error at runtime, rise an exception instead)
  return nullptr;
}

#endif // __TREE_SEARCH_H__
