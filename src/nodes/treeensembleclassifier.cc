#include "treeensembleclassifier.h"

namespace toC
{

void TreeEnsembleClassifier::parseAttributes( onnx::NodeProto &node )
{
    std::unordered_map<std::string, onnx::AttributeProto> nameToAttributeMap;
    for( const auto& a : node.attribute() )
    {
        nameToAttributeMap[a.name()] = a;
    }

    node_tree_ids = parse_attribute_ints(nameToAttributeMap["nodes_treeids"]);
    node_node_ids = parse_attribute_ints(nameToAttributeMap["nodes_nodeids"]);
    leaf_tree_ids = parse_attribute_ints(nameToAttributeMap["class_treeids"]);
    leaf_node_ids = parse_attribute_ints(nameToAttributeMap["class_nodeids"]);
    leaf_class_ids = parse_attribute_ints(nameToAttributeMap["class_ids"]);
    leaf_node_weights = parse_attribute_floats(nameToAttributeMap["class_weights"]);
    node_true_ids = parse_attribute_ints(nameToAttributeMap["nodes_truenodeids"]);
    node_false_ids = parse_attribute_ints(nameToAttributeMap["nodes_falsenodeids"]);
    node_feature_ids = parse_attribute_ints(nameToAttributeMap["nodes_featureids"]);
    node_thresholds = parse_attribute_floats(nameToAttributeMap["nodes_values"]);
    node_modes = parse_attribute_strings(nameToAttributeMap["nodes_modes"]);

    for( int i = 0; i < leaf_tree_ids.size(); ++i )
    {
        class_ids.insert(leaf_class_ids[i]);
    }
}


/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
void TreeEnsembleClassifier::resolve(void)
{
	name_input(0, "features");

	/* Create output tensors.
	 * Set data dimensions and data type for the created tensors. */
    Tensor *label_tensor = new Tensor;
    label_tensor->data_dim.push_back(1);
    label_tensor->data_type = onnx::TensorProto_DataType_INT64;

	Tensor *probabilities_tensor = new Tensor;
	probabilities_tensor->data_dim = {static_cast<int>(class_ids.size() + 1)};
	probabilities_tensor->data_type = onnx::TensorProto_DataType_FLOAT;

    // If outputs aren't registered in the same order
    // as the model outputs are listed they will be reversed.
    // Source: https://github.com/kraiskil/onnx2c/blob/855a0d7b86b1bb50aa45f0f78e4eefa6dc553085/src/graph.cc#L356
    register_output(label_tensor, "label");
	register_output(probabilities_tensor, "probabilities");
}

static std::string convertMode(std::string& mode)
{
    if( mode == "BRANCH_LEQ" )
        return "<=";
    if( mode == "BRANCH_LT" )
        return "<";
    if( mode == "BRANCH_GTE" )
        return ">=";
    if( mode == "BRANCH_GT" )
        return ">";
    if( mode == "BRANCH_EQ" )
        return "==";
    if( mode == "BRANCH_NEQ" )
        return "!=";
    if( mode == "LEAF" )
        ERROR("Attempt to convert LEAF mode in TreeEnsembleClassifier");
    ERROR("Unknown mode in TreeEnsembleClassifier");
};

std::unordered_map<int64_t, TreeEnsembleClassifier::Tree> TreeEnsembleClassifier::generateTreeMap() const
{
    std::unordered_map<int64_t, Tree> trees;

    for( int i = 0; i < node_tree_ids.size(); ++i )
    {
        int64_t treeId = node_tree_ids[i];
        int64_t nodeId = node_node_ids[i];
        trees[treeId][nodeId] = std::make_shared<TreeNode>();
    }

    for( int i = 0; i < leaf_tree_ids.size(); ++i )
    {
        int64_t treeId = leaf_tree_ids[i];
        int64_t leafId = leaf_node_ids[i];
        trees[treeId][leafId]->is_leaf = true;
        trees[treeId][leafId]->class_id = leaf_class_ids[i];
        trees[treeId][leafId]->weight = leaf_node_weights[i];
    }

    // This part must be done separate from creating the TreeNodes
    // so they all exist before connections start being made.
    for( int i = 0; i < node_tree_ids.size(); ++i )
    {
        int64_t treeId = node_tree_ids[i];
        int64_t nodeId = node_node_ids[i];
        trees[treeId][nodeId]->child_true = trees[treeId][node_true_ids[i]];
        trees[treeId][nodeId]->child_false = trees[treeId][node_false_ids[i]];
        trees[treeId][nodeId]->feature = node_feature_ids[i];
        trees[treeId][nodeId]->threshold = node_thresholds[i];
        trees[treeId][nodeId]->mode = node_modes[i];
    }

    return trees;
}

/* Body of the node implementing function */
void TreeEnsembleClassifier::print(std::ostream &dst) const
{
    std::unordered_map<int64_t, Tree> trees = generateTreeMap();

    for( int64_t class_id : class_ids )
    {
        INDT_1 << "float result_class_" << class_id << " = 0.0f;" << std::endl;
    }

	for( auto& [tree_id, tree] : trees )
    {
        INDT_1 << "/* Tree " << tree_id << " */" << std::endl;
        std::stack<std::shared_ptr<TreeNode>> node_stack;
        node_stack.push(tree.at(0));
        std::shared_ptr<TreeNode> current_node = tree.at(0);

        do
        {
            if( !current_node )
            {
                current_node = node_stack.top();
                node_stack.pop();
            }

            if( current_node->is_leaf )
            {
                INDT(node_stack.size()) << "result_class_" << current_node->class_id << " += " << current_node->weight << ";" << std::endl;
                current_node.reset();
            }
            else
            {
                node_stack.push(current_node);
                std::shared_ptr<TreeNode> nextNode;
                if (current_node->child_true)
                {
                    // features[0] because the batch size is always 1
                    INDT(node_stack.size() - 1) << "if( features[0][" << current_node->feature << "] " << convertMode(current_node->mode) << " " << current_node->threshold << " ) {" << std::endl;
                    nextNode = current_node->child_true;
                    current_node->child_true.reset();
                }
                else if (current_node->child_false)
                {
                    INDT(node_stack.size() - 1) << "} else {" << std::endl;
                    nextNode = current_node->child_false;
                    current_node->child_false.reset();
                }
                else
                {
                    if (node_stack.size() > 1)
                    {
                        INDT(node_stack.size() - 1) << "}" << std::endl;
                    }
                    nextNode.reset();
                    node_stack.pop();
                }
                current_node = nextNode;
            }
        }
        while( !node_stack.empty() );
    }

    for( int64_t classId : class_ids )
    {
        INDT_1 << "probabilities[" << classId << "] = 1.0f / (1.0f + exp(result_class_" << classId << "));" << std::endl;
    }

    INDT_1 << "probabilities[" << class_ids.size() << "] = 1.0f - (";
    size_t classIndex = 0;
    for( int64_t classId : class_ids )
    {
        dst << "probabilities[" << classId << std::string("]") + (classIndex++ < class_ids.size() - 1 ? " + " : "");
    }
    dst << ");" << std::endl;

    INDT_1 << "int64_t max_class = 0;" << std::endl;
    INDT_1 << "for( int64_t i = 0; i < " << class_ids.size() << "; i++ ) {" << std::endl;
    INDT_2 << "if( probabilities[i] > probabilities[max_class] ) {" << std::endl;
    INDT_3 << "max_class = i;" << std::endl;
    INDT_2 << "}" << std::endl;
    INDT_1 << "}" << std::endl;
    INDT_1 << "label[0] = max_class;" << std::endl;
}

}
