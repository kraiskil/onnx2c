#include "lstm.h"
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
namespace toC {


void LSTM::parseAttributes( onnx::NodeProto &node ) {
	for( const auto& a : node.attribute() ) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;

		if( a.name() == "activation_alpha" )
			activation_alpha = parse_attribute_floats(a);
		else if( a.name() == "activation_beta" )
			activation_beta = parse_attribute_floats(a);
		else if( a.name() == "activations" )
			activations = parse_attribute_strings(a);
		else if( a.name() == "clip" )
			clip = parse_attribute_float(a);
		else if( a.name() == "direction" ) {
			direction = parse_attribute_string(a);
			if( direction == "" ) direction="forward";
			else if(  direction != "forward"
			        &&direction != "reverse"
			        &&direction != "bidirectional")
				ERROR("Bad value ("<<direction<<") for direction attribute");
			// The specification is not quite clear - need test case
			if( direction == "reverse" )
				LOG(WARNING) << "Reverse LSTM might be buggy"<< std::flush;
		}
		else if( a.name() == "hidden_size" )
			hidden_size = parse_attribute_int(a);
		else if( a.name() == "input_forget" )
			input_forget = parse_attribute_int(a);
		else if( a.name() == "layout" )
			layout = parse_attribute_int(a);
		else
			ERROR("Bad attribute " << a.name() << " for LSTM");
	}
}

float LSTM::get_activation_alpha( const std::string &a)
{
	/* These activations don't have an alpha */
	if( a == "Sigmoid" )
		return 0;
	if( a == "Tanh" )
		return 0;
	if( a == "Relu" )
		return 0;

	ERROR("Unhandled: alpha for activation: " << a);
}
float LSTM::get_activation_beta( const std::string &a)
{
	/* These activations don't have a beta */
	if( a == "Sigmoid" )
		return 0;
	if( a == "Tanh" )
		return 0;
	if( a == "Relu" )
		return 0;

	ERROR("Unhandled: beta for activation: " << a);
}

void LSTM::print_activation(std::ostream &dst, const std::string &activation, const std::string &var) const
{
	std::string variable;
	if( clip < 0 )
		variable=var;
	else
		variable="CLIP(" + var + ", " + std::to_string(clip) + ")";

	if( activation == "Sigmoid" )
		dst << "1.0f/(1+expf(-" << variable << "));" << std::endl;
	else if (activation == "Tanh" )
		// TODO: optimize to tanhf? If someone uses tanh, do they care about execution speed? :)
		dst << "tanh(" << variable << ");" << std::endl;
	else if (activation == "Relu" )
		dst << "MAX(" << variable << ", 0);" << std::endl;
	else
		ERROR("Unimplmemented activation function");
}


/* Print the C code for the core LSTM kernel, inside of the "sequences" loop.
 * The code is almost identical for forward and backwards nodes */
void LSTM::print_lstm_kernel(std::ostream &dst, bool forward) const
{
	const Tensor* B = get_B();
	const Tensor* P = get_P();

	int dir;    // direction index into tensors that separate forward and backward (W,B,Y,...)
	int f_act;  // indexes for the activation functions in activations[]
	int g_act;
	int h_act;
	std::string di;
	std::string X_sbi;  // input data, indexed with s(equence), b(atch), i(input)
	std::string Yh_dbh; // Y_h, indexed with direction, batch, hidden
	std::string Yh_dbk; // Same, but use k for indexing hidden size (used in inner matmul loops)
	std::string Yc_dbh; // Y_h, indexed with direction, batch, hidden
	std::string Y_snbh; // Y, indexed with sequence, numdir, batch, hidden
	if( forward ) {
		dir=0;
		f_act=0;
		g_act=1;
		h_act=2;
		di="i";
	}
	else {
		dir=1;
		f_act=3;
		g_act=4;
		h_act=5;
		di="ds-1-i";
	}

	if( layout == 0 ) {
		X_sbi = "X[s][b]["+di+"]";
		Y_snbh = "Y[s][" + std::to_string(dir) + "][b][h]";
		Yh_dbh = "Y_h[" + std::to_string(dir) + "][b][h]";
		Yh_dbk = "Y_h[" + std::to_string(dir) + "][b][k]";
		Yc_dbh = "Y_c[" + std::to_string(dir) + "][b][h]";
	}
	else { //layout==1
		X_sbi = "X[b][s]["+di+"]";
		Y_snbh = "Y[b][s][" + std::to_string(dir) + "][h]";
		Yh_dbh = "Y_h[b][" + std::to_string(dir) + "][h]";
		Yh_dbk = "Y_h[b][" + std::to_string(dir) + "][k]";
		Yc_dbh = "Y_c[b][" + std::to_string(dir) + "][h]";
	}



	/* With all the helper strings above, print out the kernel.
	 * indexes:
	 * - b: batch size
	 * - h: hidden size
	 * - i: data size
	 * - k: hidden size, when it disappears as the inner dimension in a multiplication
	 */
	INDT_2<<  "for( int b=0; b<bs; b++)" << std::endl;
	INDT_2<<  "for( int h=0; h<hs; h++) {" << std::endl;
	INDT_3<<  "ft[b][h]=0;" << std::endl;
	INDT_3<<  "it[b][h]=0;" << std::endl;
	INDT_3<<  "ct[b][h]=0;" << std::endl;

	// Xt*W
	INDT_3<<  "for( int i=0; i<ds; i++) {" << std::endl;
	INDT_4<<  "ft[b][h] += " << X_sbi << "*W["<<dir<<"][fidx+h][i];" << std::endl;
	INDT_4<<  "it[b][h] += " << X_sbi << "*W["<<dir<<"][iidx+h][i];" << std::endl;
	INDT_4<<  "ct[b][h] += " << X_sbi << "*W["<<dir<<"][cidx+h][i];" << std::endl;
	INDT_3<<  "}" << std::endl;

	// Ht-1*R
	INDT_3<<  "for( int k=0; k<hs; k++) {" << std::endl;
	INDT_4<<  "ft[b][h] += " << Yh_dbk << "*R["<<dir<<"][fidx+h][k];" << std::endl;
	INDT_4<<  "ct[b][h] += " << Yh_dbk << "*R["<<dir<<"][cidx+h][k];" << std::endl;
	INDT_4<<  "it[b][h] += " << Yh_dbk << "*R["<<dir<<"][iidx+h][k];" << std::endl;
	INDT_3<<  "}" << std::endl;

	if( B ) { // Bias
	INDT_3<<  "ft[b][h] += B["<<dir<<"][fidx+h];" << std::endl;
	INDT_3<<  "ft[b][h] += B["<<dir<<"][Rb+fidx+h];" << std::endl;
	INDT_3<<  "it[b][h] += B["<<dir<<"][iidx+h];" << std::endl;
	INDT_3<<  "it[b][h] += B["<<dir<<"][Rb+iidx+h];" << std::endl;
	INDT_3<<  "ct[b][h] += B["<<dir<<"][cidx+h];" << std::endl;
	INDT_3<<  "ct[b][h] += B["<<dir<<"][Rb+cidx+h];" << std::endl;
	}
	if( P ) { // Peephole
	INDT_3<<  "ft[b][h] += P["<<dir<<"][fidx+h]*" << Yc_dbh << ";" << std::endl;
	INDT_3<<  "it[b][h] += P["<<dir<<"][iidx+h]*" << Yc_dbh << ";" << std::endl;
	// Cell gate does not have a peephole
	}

	// Activations
	INDT_3<<  "ft[b][h] =";
	print_activation( dst, activations[f_act], "ft[b][h]");
	INDT_3<<  "it[b][h] =";
	print_activation( dst, activations[f_act], "it[b][h]");
	INDT_3<<  "ct[b][h] =";
	print_activation( dst, activations[g_act], "ct[b][h]");
	INDT_2<< "}" << std::endl;

	// Cell state, Output gate
	INDT_2<<  "for( int b=0; b<bs; b++)" << std::endl;
	INDT_2<<  "for( int h=0; h<hs; h++) {" << std::endl;
	INDT_3<<  "/* Cell state */" << std::endl;
	INDT_3<<  Yc_dbh << " = " << Yc_dbh << "*ft[b][h] + it[b][h]*ct[b][h];" << std::endl;
	INDT_3<<  "/* Output gate */" << std::endl;
	INDT_3<<  "ot[b][h]=0;" << std::endl;
	// X*W
	INDT_3<<  "for( int i=0; i<ds; i++)" << std::endl;
	INDT_4<<  "ot[b][h] += " << X_sbi << "*W["<<dir<<"][oidx+h][i];" << std::endl;
	// Ht-1*R
	INDT_3<<  "for( int k=0; k<hs; k++)" << std::endl;
	INDT_4<<  "ot[b][h] += " << Yh_dbk << "*R["<<dir<<"][oidx+h][k];" << std::endl;
	if( B ) {// Bias
	INDT_3<<  "ot[b][h] += B["<<dir<<"][oidx+h];" << std::endl;
	INDT_3<<  "ot[b][h] += B["<<dir<<"][Rb+oidx+h];" << std::endl;
	}
	if( P ) // Peephole
	INDT_3<<  "ot[b][h] += P["<<dir<<"][oidx+h]*" << Yc_dbh << ";" << std::endl;
	INDT_3<<  "ot[b][h] =";
	print_activation( dst, activations[f_act], "ot[b][h]");
	INDT_2<<  "}" << std::endl;

	// Hidden state
	INDT_2<<  "/* Hidden state */" << std::endl;
	INDT_2<<  "for( int b=0; b<bs; b++)" << std::endl;
	INDT_2<<  "for( int h=0; h<hs; h++) {" << std::endl;
		INDT_3<< Yh_dbh << " = ot[b][h] * ";
			print_activation( dst, activations[h_act], Yc_dbh );
		if( get_Y()->is_used() ) {
			INDT_3<< Y_snbh << "= " << Yh_dbh <<";" << std::endl;
		}
	INDT_2<<  "}" << std::endl << std::endl;
}

void LSTM::print(std::ostream &dst) const
{
	const Tensor* X = get_X();
	const Tensor* W = get_W();
	const Tensor* R = get_R();
	const Tensor* B = get_B();
	const Tensor* sequence_lens = get_sequence_lens();
	const Tensor* initial_h = get_initial_h();
	const Tensor* initial_c = get_initial_c();
	const Tensor* P = get_P();

	INDT_1<< "/* LSTM " << std::endl;
	INDT_1<< " * inputs: " << std::endl;
	INDT_1<< " *   X = " << X->cname() << std::endl;
	INDT_1<< " *   W = " << W->cname() << std::endl;
	INDT_1<< " *   R = " << R->cname() << std::endl;
	INDT_1<< " *   B = " << (B?B->cname():"") << std::endl;
	INDT_1<< " *   sequence_lens = " << (sequence_lens?sequence_lens->cname():"") << std::endl;
	INDT_1<< " *   initial_h = " << (initial_h?initial_h->cname():"") << std::endl;
	INDT_1<< " *   initial_c = " << (initial_c?initial_c->cname():"") << std::endl;
	INDT_1<< " *   P = " << (P?P->cname():"") << std::endl;
	INDT_1<< " * outputs: " << std::endl;
	INDT_1<< " *   Y = " << get_Y()->cname() << std::endl;
	INDT_1<< " *   Y_h = " << get_Y_h()->cname() << std::endl;
	INDT_1<< " *   Y_c = " << get_Y_c()->cname() << std::endl;
	INDT_1<< " * attributes:" << std::endl;
	INDT_1<< " *   activations: ";
		for( auto a : activations )
			dst << a << " ";
		dst << std::endl;
	INDT_1<< " * clip: " << (clip > 0 ? std::to_string(clip) : "off") << std::endl;
	INDT_1<< " * layout: " << layout << std::endl;
	INDT_1<< " * (rest TBD):" << std::endl;
	INDT_1<< " */" << std::endl;

	const std::string data_type = X->data_type_str();
	// shorthands for code brevity
	int hs = hidden_size;
	int ds = input_size;
	int bs = batch_size;

	INDT_1<<  "int hs = " << hs << ";" << std::endl;
	INDT_1<<  "int ds = " << ds << ";" << std::endl;
	INDT_1<<  "int bs = " << bs << ";" << std::endl;
	// index into W, R to get the start of the gate indices
	INDT_1<<  "int iidx = 0;" << std::endl;
	INDT_1<<  "int oidx = hs;" << std::endl;
	INDT_1<<  "int fidx = 2*hs;" << std::endl;
	INDT_1<<  "int cidx = 3*hs;" << std::endl;
	// index into B, to get Rb. Wb is B at offset 0
	INDT_1<<  "int Rb = 4*hs;" << std::endl;
	// TODO: variable lenght sequences not yet implemented
	INDT_1<<  "int sequence_lenght = " << seq_length << ";" << std::endl;

	// TODO: these temporary variables are BIG. Make them global to minimize
	// stack usage? Probably needs to be an onnx2c flag for user to select
	INDT_1<<  "/* Forget gate */" << std::endl;
	INDT_1<<  data_type << " ft[bs][hs];" << std::endl;
	INDT_1<<  "/* Input gate */" << std::endl;
	INDT_1<<  data_type << " it[bs][hs];" << std::endl;
	INDT_1<<  "/* Cell gate */" << std::endl;
	INDT_1<<  data_type << " ct[bs][hs];" << std::endl;
	INDT_1<<  "/* Output gate */" << std::endl;
	INDT_1<<  data_type << " ot[bs][hs];" << std::endl;
	dst << std::endl;

	/* Initialize cell and hidden state at the start of a run.
	 * TODO: should these not be reset at the start of each sequence run?
	 *       The documentation doesn't say, but this passes backend tests...
	 */
	if( initial_h && initial_h->is_used() )
		INDT_1 << "memcpy(Y_h, initial_h, sizeof(*initial_h));" << std::endl;
	else
		INDT_1 << "memset(Y_h, 0, sizeof(*Y_h));" << std::endl;
	if( initial_c && initial_c->is_used() )
		INDT_1 << "memcpy(Y_c, initial_c, sizeof(*initial_c));" << std::endl;
	else
		INDT_1 << "memset(Y_c, 0, sizeof(*Y_c));" << std::endl;
	dst << std::endl;

	/* Loop over sequences */
	INDT_1<<  "for( int s=0; s<sequence_lenght; s++) {" << std::endl;
	dst << std::endl;
	INDT_2<<  "/* Forward lane */" << std::endl;
	print_lstm_kernel(dst, /* forward= */ true);

	if( direction == "bidirectional" ) {
		dst << std::endl;
		INDT_2<<  "/* Backward lane */" << std::endl;
		print_lstm_kernel(dst, /* forward= */ false);
	}
	INDT_1<<  "} /* sequences */" << std::endl;

}


// Helper function for resolve(void)
void LSTM::calculate_data_dimensions()
{
	const Tensor* X = get_X();
	const Tensor* W = get_W();
	if( layout == 0 ) {
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

void LSTM::resolve(void)
{
	if( inputs.size() < 3 || inputs.size() > 8 )
		ERROR("wrong number of inputs to LSTM");


	// Set attribute default values for those attributes that are not set in the model
	if( activations.size() == 0 ) {
		activations.push_back("Sigmoid");
		activations.push_back("Tanh");
		activations.push_back("Tanh");
		if( direction == "bidirectional" ) {
			activations.push_back("Sigmoid");
			activations.push_back("Tanh");
			activations.push_back("Tanh");
		}
	}
	if( activations.size() != 3 && activations.size() != 6)
		ERROR("Error - bad number of activations attributes");

	if( activation_alpha.size() == 0 ) {
		for( auto &a : activations ) {
			activation_alpha.push_back(get_activation_alpha(a));
		}
	}
	if( activation_alpha.size() != 3 && activation_alpha.size() != 6)
		ERROR("Unimplemented/error: not 3(6) activation alphas");

	if( activation_beta.size() == 0 ) {
		for( auto &a : activations ) {
			activation_beta.push_back(get_activation_beta(a));
		}
	}
	if( activation_beta.size() != 3 && activation_beta.size() != 6)
		ERROR("Unimplemented/error: not 3(6) activation betas");

	if( hidden_size < 0 )
		ERROR("Must provide hidden_size attribute!");

	register_input(get_X(), "X");
	register_input(get_W(), "W");
	register_input(get_R(), "R");

	//optional inputs. Trailing unprovided inputs can just be left out
	//but non-trailing, unprovided inputs MUST have an empty string as name
	// (guess that means tensors MAY NOT have an empty string as name?)
	if( get_B() ) {
		register_input(get_B(), "B");
	}
	if( get_sequence_lens() ) {
		register_input(get_sequence_lens(), "sequence_lens");
	}
	if( get_initial_h()) {
		register_input(get_initial_h(), "initial_h");
	}
	if( get_initial_c()) {
		register_input(get_initial_c(), "initial_c");
	}
	if( get_P() ) {
		register_input(get_P(), "P");
	}


	calculate_data_dimensions();

	if( get_sequence_lens() ) {
		const Tensor* sequence_lens = get_sequence_lens();

		if( static_cast<int>(sequence_lens->rank()) != 1 )
			ERROR("If providing sequence lengths, it must be a 1D tensor");
		if( static_cast<int>(sequence_lens->data_dim[0]) != batch_size )
			ERROR("If providing sequence lengths, there must be 'batch_size' of them");
		for( auto sl : sequence_lens->data_dim )
			if( sl < seq_length )
				// Not quite sure if I understand the documentation correctly here.
				ERROR("Error: requested sequence lenght is longer than input data");
	}


	// Generate output tensors.

	Tensor *Y = new Tensor;
	Y->data_type = get_X()->data_type;
	std::vector<int> y_size;
	if( layout == 0 )
		y_size = std::vector<int>({ seq_length, num_directions, batch_size, hidden_size });
	else
		y_size = std::vector<int>({ batch_size, seq_length, num_directions, hidden_size });
	Y->data_dim = y_size;

	// Y_h and Y_c are special: optional as outputs to the rest of the network,
	// but mandatory as outputs to this node itself.
	std::vector<int> ych_size;
	if( layout == 0 )
		ych_size = std::vector<int>({ num_directions, batch_size, hidden_size });
	else
		ych_size = std::vector<int>({ batch_size, num_directions, hidden_size });

	Tensor *Y_h = new Tensor;
	Y_h->data_type = get_X()->data_type;
	Y_h->data_dim = ych_size;
	Y_h->isRecursive=true;
	Y_h->data_buffer = calloc(Y_h->data_num_elem(), Y_h->data_elem_size());
	if( Y_h->data_buffer == NULL )
		ERROR("Memory allocation failed");
	Y_h->initialize = true;

	Tensor *Y_c = new Tensor;
	Y_c->data_type = get_X()->data_type;
	Y_c->data_dim = ych_size;
	Y_c->isRecursive=true;
	Y_c->data_buffer = calloc(Y_c->data_num_elem(), Y_c->data_elem_size());
	if( Y_c->data_buffer == NULL )
		ERROR("Memory allocation failed");
	Y_c->initialize = true;

	register_output(Y, "Y");
	register_output(Y_h, "Y_h");
	register_output(Y_c, "Y_c");
}

}

