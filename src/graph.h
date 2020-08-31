
#include "onnx.pb.h"

#include "node.h"
#include "tensor.h"

namespace toC {

class Graph {
public:
	Graph(
		onnx::ModelProto &onnx_model,
		bool verbose_mode = false,
		std::vector<Tensor*> inputs={}
	);

	/* print the entire .h and .cc file contents */
	void print_header(std::ostream &destination);
	void print_source(std::ostream &destination);

	/* print individual parts of the file */
	void print_file_frontmatter(std::ostream &destination);
	void print_global_tensors(std::ostream &destination);
	void print_tensor(std::ostream &dst, const Tensor *t);
	void print_functions(std::ostream &destination);
	void print_includes(std::ostream &dst);
	void print_interface_function(std::ostream &dst);

	/* Add already resolved onnx::TensorProto. E.g. TensorProtos that
	 * are resolved already in the ONNX model (inputs and initialized ones)
	 */
	void addResolvedTensor(onnx::TensorProto &tensor);
	Tensor* getIoTensor(onnx::ValueInfoProto &vi);

	void tryResolveNode(onnx::NodeProto &node);
	bool hasUnresolvedNodes(void);
	bool nodeInputsResolved(const onnx::NodeProto &node, std::vector<const Tensor*> &inputs);
	Node* findNode(std::string opName);

private:
	onnx::ModelProto &model;
	std::vector<Tensor*> tensors;
	std::vector<Node*> nodes;
	bool verbose_mode;

	/* Add new tensor to set of known tensors,
	 * if it is not already there (checked by name) */
	bool addTensor(Tensor *t);

	Tensor *findTensor(const std::string &name) const;

	static int anonymous_nodes;
};

}

