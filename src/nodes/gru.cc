#include "gru.h"
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
namespace toC {

void GRU::parseAttributes(onnx::NodeProto& node)
{
	for (const auto& a : node.attribute()) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;

		if (a.name() == "activation_alpha")
			activation_alpha = parse_attribute_floats(a);
		else if (a.name() == "activation_beta")
			activation_beta = parse_attribute_floats(a);
		else if (a.name() == "activations")
			activations = parse_attribute_strings(a);
		else if (a.name() == "clip")
			clip = parse_attribute_float(a);
		else if (a.name() == "direction") {
			direction = parse_attribute_string(a);
			if (direction == "")
				direction = "forward";
			else if (direction != "forward" && direction != "reverse" && direction != "bidirectional")
				ERROR("Bad value (" << direction << ") for direction attribute");
		}
		else if (a.name() == "hidden_size")
			hidden_size = parse_attribute_int(a);
		else if (a.name() == "linear_before_reset")
			linear_before_reset = parse_attribute_int(a);
		else if (a.name() == "layout")
			layout = parse_attribute_int(a);
		else
			ERROR("Bad attribute " << a.name() << " for GRU");
	}
}

float GRU::get_activation_alpha(const std::string& a)
{
	/* These activations don't have an alpha */
	if (a == "Sigmoid")
		return 0;
	if (a == "Tanh")
		return 0;
	if (a == "Relu")
		return 0;

	ERROR("Unhandled: alpha for activation: " << a);
}
float GRU::get_activation_beta(const std::string& a)
{
	/* These activations don't have a beta */
	if (a == "Sigmoid")
		return 0;
	if (a == "Tanh")
		return 0;
	if (a == "Relu")
		return 0;

	ERROR("Unhandled: beta for activation: " << a);
}

void GRU::print_activation(std::ostream& dst, const std::string& activation, const std::string& var) const
{
	std::string variable;
	if (clip < 0)
		variable = var;
	else
		variable = "CLIP(" + var + ", " + std::to_string(clip) + ")";

	if (activation == "Sigmoid")
		dst << "1.0f/(1+" << math_func("exp") << "(-" << variable << "));" << std::endl;
	else if (activation == "Tanh")
		dst << math_func("tanh") << "(" << variable << ");" << std::endl;
	else if (activation == "Relu")
		dst << "MAX(" << variable << ", 0);" << std::endl;
	else
		ERROR("Unimplmemented activation function");
}

/* Print the C code for the core GRU kernel, inside of the "sequences" loop.
 * The code is almost identical for forward and backward nodes */
void GRU::print_gru_kernel(std::ostream& dst, bool forward) const
{
	const Tensor* B = get_B();

	// The backward lane only has its own direction/activation index
	// in a bidirectional node. A direction=reverse node is unidirectional:
	// it uses the (single) direction 0 and the first pair of activations.
	bool backward_lane = !forward && direction == "bidirectional";
	int dir;   // direction index into tensors that separate forward and backward (W,B,Y,...)
	int f_act; // indexes for the activation functions in activations[]
	int g_act;
	std::string di;  // sequence index expression: input index for this direction
	std::string X_sbi;  // input data, indexed with s(equence), b(atch), i(input)
	std::string Yh_dbh; // Y_h, indexed with direction, batch, hidden
	std::string Yh_dbk; // Same, but use k for indexing hidden size (used in inner matmul loops)
	std::string Y_snbh; // Y, indexed with sequence, numdir, batch, hidden
	if (backward_lane) {
		dir = 1;
		f_act = 2;
		g_act = 3;
	}
	else {
		dir = 0;
		f_act = 0;
		g_act = 1;
	}
	di = forward ? "s" : "sequence_length-1-s";

	if (layout == 0) {
		X_sbi = "X[" + di + "][b][i]";
		Y_snbh = "Y[" + di + "][" + std::to_string(dir) + "][b][h]";
		Yh_dbh = "Y_h[" + std::to_string(dir) + "][b][h]";
		Yh_dbk = "Y_h[" + std::to_string(dir) + "][b][k]";
	}
	else { //	layout==1
		X_sbi = "X[b][" + di + "][i]";
		Y_snbh = "Y[b][" + di + "][" + std::to_string(dir) + "][h]";
		Yh_dbh = "Y_h[b][" + std::to_string(dir) + "][h]";
		Yh_dbk = "Y_h[b][" + std::to_string(dir) + "][k]";
	}

	/* With the helper strings above, print out the kernel.
	 * indexes:
	 * - b: batch size
	 * - h: hidden size
	 * - i: input data size
	 * - k: hidden size, when it disappears as the inner dimension in a multiplication
	 *
	 * Gate order in W and R is: update gate z, reset gate r, candidate h.
	 */
	INDT_2 << "/* Update and reset gates */" << std::endl;
	INDT_2 << "for( int b=0; b<bs; b++)" << std::endl;
	INDT_2 << "for( int h=0; h<hs; h++) {" << std::endl;
	INDT_3 << "zt[b][h]=0;" << std::endl;
	INDT_3 << "rt[b][h]=0;" << std::endl;

	// Xt*W
	INDT_3 << "for( int i=0; i<ds; i++) {" << std::endl;
	INDT_4 << "zt[b][h] += " << X_sbi << "*W[" << dir << "][zidx+h][i];" << std::endl;
	INDT_4 << "rt[b][h] += " << X_sbi << "*W[" << dir << "][ridx+h][i];" << std::endl;
	INDT_3 << "}" << std::endl;

	// Ht-1*R
	INDT_3 << "for( int k=0; k<hs; k++) {" << std::endl;
	INDT_4 << "zt[b][h] += " << Yh_dbk << "*R[" << dir << "][zidx+h][k];" << std::endl;
	INDT_4 << "rt[b][h] += " << Yh_dbk << "*R[" << dir << "][ridx+h][k];" << std::endl;
	INDT_3 << "}" << std::endl;

	if (B) { // Bias
		INDT_3 << "zt[b][h] += B[" << dir << "][zidx+h];" << std::endl;
		INDT_3 << "zt[b][h] += B[" << dir << "][Rb+zidx+h];" << std::endl;
		INDT_3 << "rt[b][h] += B[" << dir << "][ridx+h];" << std::endl;
		INDT_3 << "rt[b][h] += B[" << dir << "][Rb+ridx+h];" << std::endl;
	}

	// Activations
	INDT_3 << "zt[b][h] =";
	print_activation(dst, activations[f_act], "zt[b][h]");
	INDT_3 << "rt[b][h] =";
	print_activation(dst, activations[f_act], "rt[b][h]");
	INDT_2 << "}" << std::endl;

	// Candidate hidden state
	INDT_2 << "/* Candidate hidden state */" << std::endl;
	INDT_2 << "for( int b=0; b<bs; b++)" << std::endl;
	INDT_2 << "for( int h=0; h<hs; h++) {" << std::endl;
	INDT_3 << "nt[b][h]=0;" << std::endl;
	// Xt*W
	INDT_3 << "for( int i=0; i<ds; i++)" << std::endl;
	INDT_4 << "nt[b][h] += " << X_sbi << "*W[" << dir << "][hidx+h][i];" << std::endl;
	if (linear_before_reset == 0) {
		// Ht-1*R, with the reset gate applied to Ht-1 before the multiplication
		INDT_3 << "for( int k=0; k<hs; k++)" << std::endl;
		INDT_4 << "nt[b][h] += rt[b][k]*" << Yh_dbk << "*R[" << dir << "][hidx+h][k];" << std::endl;
	}
	else {
		// Ht-1*R first, reset gate applied to the whole sum
		INDT_3 << "float pre=0;" << std::endl;
		INDT_3 << "for( int k=0; k<hs; k++)" << std::endl;
		INDT_4 << "pre += " << Yh_dbk << "*R[" << dir << "][hidx+h][k];" << std::endl;
		if (B)
			INDT_3 << "pre += B[" << dir << "][Rb+hidx+h];" << std::endl;
		INDT_3 << "nt[b][h] += rt[b][h]*pre;" << std::endl;
	}
	if (B) { // Bias (the R-bias is already applied above for linear_before_reset)
		INDT_3 << "nt[b][h] += B[" << dir << "][hidx+h];" << std::endl;
		if (linear_before_reset == 0)
			INDT_3 << "nt[b][h] += B[" << dir << "][Rb+hidx+h];" << std::endl;
	}
	INDT_3 << "nt[b][h] =";
	print_activation(dst, activations[g_act], "nt[b][h]");
	INDT_2 << "}" << std::endl;

	// Hidden state update: Ht = (1-Zt)*Ht~ + Zt*Ht-1
	INDT_2 << "/* Hidden state */" << std::endl;
	INDT_2 << "for( int b=0; b<bs; b++)" << std::endl;
	INDT_2 << "for( int h=0; h<hs; h++) {" << std::endl;
	INDT_3 << Yh_dbh << " = (1-zt[b][h])*nt[b][h] + zt[b][h]*" << Yh_dbh << ";" << std::endl;
	if (get_Y()->is_used()) {
		INDT_3 << Y_snbh << "= " << Yh_dbh << ";" << std::endl;
	}
	INDT_2 << "}" << std::endl
	       << std::endl;
}

void GRU::print(std::ostream& dst) const
{
	const Tensor* X = get_X();
	const Tensor* W = get_W();
	const Tensor* R = get_R();
	const Tensor* B = get_B();
	const Tensor* sequence_lens = get_sequence_lens();
	const Tensor* initial_h = get_initial_h();

	INDT_1 << "/* GRU " << std::endl;
	INDT_1 << " * inputs: " << std::endl;
	INDT_1 << " *   X = " << X->cname() << std::endl;
	INDT_1 << " *   W = " << W->cname() << std::endl;
	INDT_1 << " *   R = " << R->cname() << std::endl;
	INDT_1 << " *   B = " << (B ? B->cname() : "") << std::endl;
	INDT_1 << " *   sequence_lens = " << (sequence_lens ? sequence_lens->cname() : "") << std::endl;
	INDT_1 << " *   initial_h = " << (initial_h ? initial_h->cname() : "") << std::endl;
	INDT_1 << " * outputs: " << std::endl;
	INDT_1 << " *   Y = " << get_Y()->cname() << std::endl;
	INDT_1 << " *   Y_h = " << get_Y_h()->cname() << std::endl;
	INDT_1 << " * attributes:" << std::endl;
	INDT_1 << " *   activations: ";
	for (auto a : activations)
		dst << a << " ";
	dst << std::endl;
	INDT_1 << " * clip: " << (clip > 0 ? std::to_string(clip) : "off") << std::endl;
	INDT_1 << " * direction: " << direction << std::endl;
	INDT_1 << " * layout: " << layout << std::endl;
	INDT_1 << " * linear_before_reset: " << linear_before_reset << std::endl;
	INDT_1 << " */" << std::endl;

	const std::string data_type = X->data_type_str();
	// shorthands for code brevity
	int hs = hidden_size;
	int ds = input_size;
	int bs = batch_size;

	INDT_1 << "int hs = " << hs << ";" << std::endl;
	INDT_1 << "int ds = " << ds << ";" << std::endl;
	INDT_1 << "int bs = " << bs << ";" << std::endl;
	// index into W, R to get the start of the gate indices
	INDT_1 << "int zidx = 0;" << std::endl;
	INDT_1 << "int ridx = hs;" << std::endl;
	INDT_1 << "int hidx = 2*hs;" << std::endl;
	if (B) {
		// index into B, to get Rb. Wb is B at offset 0
		INDT_1 << "int Rb = 3*hs;" << std::endl;
	}
	INDT_1 << "int sequence_length = " << seq_length << ";" << std::endl;

	INDT_1 << "/* Update gate */" << std::endl;
	INDT_1 << data_type << " zt[" << bs << "][" << hs << "];" << std::endl;
	INDT_1 << "/* Reset gate */" << std::endl;
	INDT_1 << data_type << " rt[" << bs << "][" << hs << "];" << std::endl;
	INDT_1 << "/* Candidate hidden state */" << std::endl;
	INDT_1 << data_type << " nt[" << bs << "][" << hs << "];" << std::endl;
	dst << std::endl;

	// Initialize the hidden state at the start of a run.
	// NB: must copy/reset the whole tensor, not just the first
	// direction slice (the array-parameter sizeof() only covers [dir][...])
	int yh_bytes = get_Y_h()->data_num_elem();
	if (initial_h && initial_h->is_used())
		INDT_1 << "memcpy(Y_h, initial_h, " << yh_bytes << "*sizeof(" << data_type << "));" << std::endl;
	else
		INDT_1 << "memset(Y_h, 0, " << yh_bytes << "*sizeof(" << data_type << "));" << std::endl;
	dst << std::endl;

	/* Loop over sequences */
	INDT_1 << "for( int s=0; s<sequence_length; s++) {" << std::endl;
	dst << std::endl;
	if (direction == "reverse") {
		INDT_2 << "/* Reverse lane */" << std::endl;
		print_gru_kernel(dst, /* forward= */ false);
	}
	else {
		INDT_2 << "/* Forward lane */" << std::endl;
		print_gru_kernel(dst, /* forward= */ true);
	}

	if (direction == "bidirectional") {
		dst << std::endl;
		INDT_2 << "/* Backward lane */" << std::endl;
		print_gru_kernel(dst, /* forward= */ false);
	}
	INDT_1 << "} /* sequences */" << std::endl;
}

// Helper function for resolve(void)
void GRU::calculate_data_dimensions()
{
	const Tensor* X = get_X();
	const Tensor* W = get_W();
	if (layout == 0) {
		seq_length = X->data_dim[0];
		batch_size = X->data_dim[1];
		num_directions = W->data_dim[0];
	}
	else { // layout==1
		seq_length = X->data_dim[1];
		batch_size = X->data_dim[0];
		num_directions = W->data_dim[0];
	}
	input_size = X->data_dim[2];
}

void GRU::resolve(void)
{
	if (get_number_of_inputs() < 3 || get_number_of_inputs() > 6)
		ERROR("wrong number of inputs to GRU");

	// Set attribute default values for those attributes that are not set in the model
	if (activations.size() == 0) {
		activations.push_back("Sigmoid");
		activations.push_back("Tanh");
		if (direction == "bidirectional") {
			activations.push_back("Sigmoid");
			activations.push_back("Tanh");
		}
	}
	if (activations.size() != 2 && activations.size() != 4)
		ERROR("Error - bad number of activations attributes");

	if (activation_alpha.size() == 0) {
		for (auto& a : activations) {
			activation_alpha.push_back(get_activation_alpha(a));
		}
	}
	if (activation_beta.size() == 0) {
		for (auto& a : activations) {
			activation_beta.push_back(get_activation_beta(a));
		}
	}

	if (hidden_size < 0)
		ERROR("Must provide hidden_size attribute!");
	if (layout != 0 && layout != 1)
		ERROR("Bad value for layout attribute");
	if (linear_before_reset != 0 && linear_before_reset != 1)
		ERROR("Bad value for linear_before_reset attribute");

	name_input(0, "X");
	name_input(1, "W");
	name_input(2, "R");

	//	optional inputs. Trailing unprovided inputs can just be left out
	//	but non-trailing, unprovided inputs MUST have an empty string as name
	if (get_B()) {
		name_input(3, "B");
	}
	if (get_sequence_lens()) {
		name_input(4, "sequence_lens");
	}
	if (get_initial_h()) {
		name_input(5, "initial_h");
	}

	if (get_sequence_lens())
		ERROR("Unimplemented: GRU with sequence_lens (variable length sequences)");

	calculate_data_dimensions();

	// Validate tensor shapes against the node attributes/dimensions
	const Tensor* W = get_W();
	const Tensor* R = get_R();
	if (W->rank() != 3 || R->rank() != 3)
		ERROR("GRU: W and R must be 3-dimensional tensors");
	if (static_cast<int>(W->data_dim[0]) != num_directions
	    || static_cast<int>(R->data_dim[0]) != num_directions)
		ERROR("GRU: W and R must have the same number of directions");
	if (direction == "bidirectional" && num_directions != 2)
		ERROR("GRU: direction is bidirectional, but weights are not");
	if (direction != "bidirectional" && num_directions != 1)
		ERROR("GRU: direction is unidirectional, but weights are bidirectional");
	if (static_cast<int>(W->data_dim[1]) != 3 * hidden_size)
		ERROR("GRU: W must have 3*hidden_size rows");
	if (static_cast<int>(W->data_dim[2]) != input_size)
		ERROR("GRU: W input size does not match X");
	if (static_cast<int>(R->data_dim[1]) != 3 * hidden_size)
		ERROR("GRU: R must have 3*hidden_size rows");
	if (static_cast<int>(R->data_dim[2]) != hidden_size)
		ERROR("GRU: R hidden size does not match hidden_size attribute");

	const Tensor* B = get_B();
	if (B) {
		if (B->rank() != 2 || static_cast<int>(B->data_dim[1]) != 6 * hidden_size
		    || static_cast<int>(B->data_dim[0]) != num_directions)
			ERROR("GRU: B must have shape [num_directions, 6*hidden_size]");
	}

	const Tensor* initial_h = get_initial_h();
	if (initial_h) {
		std::vector<int> expected = layout == 0
			? std::vector<int>({num_directions, batch_size, hidden_size})
			: std::vector<int>({batch_size, num_directions, hidden_size});
		if (initial_h->data_dim != expected)
			ERROR("GRU: initial_h has a bad shape");
	}

	// Generate output tensors.
	Tensor* Y = new Tensor;
	Y->data_type = get_X()->data_type;
	std::vector<int> y_size;
	if (layout == 0)
		y_size = std::vector<int>({seq_length, num_directions, batch_size, hidden_size});
	else
		y_size = std::vector<int>({batch_size, seq_length, num_directions, hidden_size});
	Y->data_dim = y_size;

	// Y_h is special: optional as output to the rest of the network,
	// but mandatory as output to this node itself.
	std::vector<int> yh_size;
	if (layout == 0)
		yh_size = std::vector<int>({num_directions, batch_size, hidden_size});
	else
		yh_size = std::vector<int>({batch_size, num_directions, hidden_size});

	Tensor* Y_h = new Tensor;
	Y_h->data_type = get_X()->data_type;
	Y_h->data_dim = yh_size;
	Y_h->isRecursive = true;
	Y_h->data_buffer = calloc(Y_h->data_num_elem(), Y_h->data_elem_size());
	if (Y_h->data_buffer == NULL)
		ERROR("Memory allocation failed");
	Y_h->initialize = true;

	register_output(Y, "Y");
	register_output(Y_h, "Y_h");

	set_math_type(get_X()->data_type);
}

} // namespace toC
