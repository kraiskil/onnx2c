/* This file is part of onnx2c.
 *
 * Graph_io is an onnx2c-specific meta-node.
 * This does not have an equivalent in the ONNX graph.
 * Each graph has two of these nodes, the graph input
 * and graph output.
 * These nodes only exists for symmetry in onnx2c graph
 * traversing algorithms and optimization passes.
 *
 * These meta-nodes are a fairly late addition to onnx2c,
 * so there might be places in the code base where
 * the usage of this kind of node would make sense,
 * but is missing.
 */
#include "node.h"

namespace toC {

class graph_io : public Node {
	public:
	graph_io()
	{
		op_name = "graph_io";
	}

	// Mandatory "API" functions towards the rest of onnx2c
	virtual void parseAttributes(onnx::NodeProto& node) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream& dst) const override;
};

/* Parse attributes, if this node has them. */
void graph_io::parseAttributes(onnx::NodeProto& node)
{
	// No attributes for special nodes
}

void graph_io::resolve(void)
{
	// These special nodes are handled .. specially. In graph.cc
}

void graph_io::print(std::ostream& dst) const
{
	// nothing to do here
}

} // namespace toC
