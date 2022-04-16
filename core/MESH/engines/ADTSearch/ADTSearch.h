#ifndef __ADT_SEARCH_H__
#define __ADT_SEARCH_H__

#include <list>
#include <stack>

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

// Alternating Digital Tree implementation for tree-based search of elements over an unstructured mesh
template <unsigned int M, unsigned int N>
class ADTSearch{
private:
  Tree<ADTnode<2*N>> tree;
  Mesh<M,N>& mesh_;

  // build an Alternating Digital Tree given a set of 2N-dimensional points.
  void init(const std::vector<std::pair<SVector<2*N>, unsigned int>>& data);
  
  // performs a geometric search returning all points which lie in a given query
  std::list<unsigned int> geometricSearch(const Query<2*N>& query);
  
public:
  // initialize the ADT using informations coming from the mesh
  ADTSearch(Mesh<M,N>& mesh);
  
  // applies the ADT geometric search to return the mesh element containing a given point
  std::shared_ptr<Element<M, N>> search(const SVector<N>& point);

  // getter
  Tree<ADTnode<2*N>> getTree() const { return tree; }
};

#include "ADTSearch.tpp"

#endif // __ADT_SEARCH_H__
