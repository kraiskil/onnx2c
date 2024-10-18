/* This file is part of onnx2c.
 *
 * TreeEnsembleClassifier node.
 *
 */
#include "node.h"
#include "util.h"

#include <set>
#include <stack>
#include <stdint.h>
#include <unordered_map>

namespace toC {

class TreeEnsembleClassifier : public Node {
	public:
	TreeEnsembleClassifier() {
		op_name = "TreeEnsembleClassifier";
	}

    std::vector<int64_t> node_tree_ids;
    std::vector<int64_t> node_node_ids;
    std::vector<int64_t> leaf_tree_ids;
    std::vector<int64_t> leaf_node_ids;
    std::vector<int64_t> leaf_class_ids;
    std::vector<float> leaf_node_weights;
    std::vector<int64_t> node_true_ids;
    std::vector<int64_t> node_false_ids;
    std::vector<int64_t> node_feature_ids;
    std::vector<float> node_thresholds;
    std::vector<std::string> node_modes;
    std::set<int64_t> class_ids;

    struct TreeNode {
        int64_t id;
        bool is_leaf;
        int64_t class_id;
        float weight;
        std::shared_ptr<TreeNode> child_true;
        std::shared_ptr<TreeNode> child_false;
        int64_t feature;
        float threshold;
        std::string mode;
    };

    using Tree = std::unordered_map<int64_t, std::shared_ptr<TreeNode>>;

	// Mandatory "API" functions towards the rest of onnx2c
	virtual void parseAttributes( onnx::NodeProto &node ) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;

    private:
    std::unordered_map<int64_t, Tree> generateTreeMap() const;
};


}

