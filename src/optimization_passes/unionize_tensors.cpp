#include "graph.h"
#include <cstdint>

using namespace toC;
// add t to union. Return allocated union number
uint32_t Graph::add_to_free_union(Tensor *t)
{
	unsigned u=0;
	// Check if tensor is already allocated to an union
	for( ; u<tensor_unions.size(); u++ )
		if( tensor_unions[u] == t)
			return u;

	// If not, search for free unions
	u=0;
	for( ; u<tensor_unions.size(); u++ )
		if( tensor_unions[u] == NULL ) {
			tensor_unions[u] = t;
			t->assign_union(u);
			return u;
		}

	// All unions in use, need a new one
	LOG(TRACE) << "No free unions, creating a new one" << std::endl;
	tensor_unions.push_back(t);
	t->assign_union( tensor_unions.size()-1 );
	return t->union_no;
}
void Graph::mark_union_unoccupied(uint32_t u)
{
	tensor_unions[u]=NULL;
}
// tag intermediate (graph internal) tensors
// with union numbers so they can share memory
// in a temporal fashion
void Graph::unionize_tensors(void)
{
	for( auto n : nodes ) {
		n->isResolved = false;
	}

	for( auto n : nodes ) {

		// for each output tensor of node
		for( auto o : n->outputs ) {
			// assign tensor to next free union
			// if it is an internal tensor that gets
			// calculated by a node.
			if( o->is_used() == false )
				continue;
			if( o->isIO == true )
				continue;
			// the Constant node is a bit weird - this check must be in
			if( o->isConst == true )
				continue;
			if( o->initialize == true )
				continue;
			add_to_free_union(o);
		}
		// mark node as resolved
		n->isResolved = true;

		// Check if union slots can be re-used
		for( unsigned ui=0; ui < tensor_unions.size(); ui++ ) {
			Tensor *t = tensor_unions[ui];
			// skip free slots
			if ( t==NULL )
				continue;

			// when all the consumers of this tensors have consumed it
			// the tensor is nolonger needed, and the union is free to host
			// a new tensor
			bool all_resolved = true;
			for( auto c : t->consumers )
				all_resolved &= c->isResolved;
			if (all_resolved)
				mark_union_unoccupied(ui);
		}
	}
}

