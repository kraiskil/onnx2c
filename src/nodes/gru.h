/* This file is part of onnx2c.
 *
 * GRU node.
 * Implements a Gated Recurrent Unit.
 * The equations are given in the ONNX operator specification:
 * https://onnx.ai/onnx/operators/onnx__GRU.html
 * They match the description in:
 * https://arxiv.org/abs/1412.3555
 *
 * NB: The Y_h output must always be available, even if the network
 * marks it optional (it is the recursion tensor).
 * If the initial_h initializer is given, the Y_h tensor is initialized
 * from it, so the hidden state is saved & updated in the Y_h tensor.
 */

#include "node.h"
namespace toC {

class GRU : public Node {
	public:
	GRU()
	{
		op_name = "GRU";
		clip = -1.0;
		hidden_size = -1;
		linear_before_reset = 0;
		layout = 0;
		direction = "forward";
	}

	// Attributes
	std::vector<float> activation_alpha;
	std::vector<float> activation_beta;
	std::vector<std::string> activations; // in order: activations f, g per direction
	float clip;                           // negative for no clip
	std::string direction;
	int hidden_size;
	int linear_before_reset;
	int layout;

	// "implicit" attributes, taken from input tensor dimensions
	int seq_length;
	int batch_size;
	int num_directions;
	int input_size;

	virtual void parseAttributes(onnx::NodeProto& node) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream& dst) const override;

	float get_activation_alpha(const std::string& a);
	float get_activation_beta(const std::string& a);
	const Tensor* get_X(void) const { return get_input_tensor(0); }
	const Tensor* get_W(void) const { return get_input_tensor(1); }
	const Tensor* get_R(void) const { return get_input_tensor(2); }
	const Tensor* get_Y(void) const { return get_output_tensor(0); }
	const Tensor* get_Y_h(void) const { return get_output_tensor(1); }

	// ONNX allows omitting optional inputs by either:
	//  - not give them at all
	//  - named with the empty string
	const Tensor* get_optional(unsigned N) const
	{
		if (get_number_of_inputs() <= N)
			return nullptr;
		if (get_input_tensor(N)->name == "")
			return nullptr;
		return get_input_tensor(N);
	}
	const Tensor* get_B(void) const { return get_optional(3); }
	const Tensor* get_sequence_lens(void) const { return get_optional(4); }
	const Tensor* get_initial_h(void) const { return get_optional(5); }

	void print_activation(std::ostream& dst, const std::string& activation, const std::string& var) const;
	void print_gru_kernel(std::ostream& dst, bool forward) const;
	void calculate_data_dimensions();
};
} // namespace toC
