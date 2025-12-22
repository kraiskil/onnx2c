/* This file is part of onnx2c.
 *
 * LSTM.
 * Implements a Long Short Term Memory node.
 * A nice description is given in:
 * http://colah.github.io/posts/2015-08-Understanding-LSTMs/
 * A nice video with the maths written out:
 * https://www.youtube.com/watch?v=Opj2AT0iYCw
 * The exact equations used for ONNX's LSTM are given in the
 * specification, and the two seem to match the two links above.
 *
 * NB: Y_h and Y_c must both always be available, even if network
 * marks them optional. (Since they are the recursion tensors).
 * If the initializers initial_h or initial_c are given, the Y_[h,c]
 * tensors are aliased to the respectie one, and the LSTM hidden/cell
 * state is saved & updated in the initial_[h,c] tensor.
 */

#include "node.h"
namespace toC {

class LSTM : public Node {
	public:
	LSTM()
	{
		op_name = "LSTM";
		clip = -1.0;
		hidden_size = -1;
		input_forget = 0;
		layout = 0;
	}

	// Attributes
	std::vector<float> activation_alpha;
	std::vector<float> activation_beta;
	std::vector<std::string> activations; // in order, activations f, g, & h
	float clip;                           // negative for no clip
	std::string direction;
	int hidden_size;
	int input_forget;
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
	const Tensor* get_Y_c(void) const { return get_output_tensor(2); }

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
	const Tensor* get_initial_c(void) const { return get_optional(6); }
	const Tensor* get_P(void) const { return get_optional(7); }

	void print_activation(std::ostream& dst, const std::string& activation, const std::string& var) const;
	void print_lstm_kernel(std::ostream& dst, bool forward) const;
	void calculate_data_dimensions();
};
} // namespace toC
