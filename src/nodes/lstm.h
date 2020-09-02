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

class LSTM : public Node {
	public:
	LSTM() {
		op_name = "LSTM";
		clip = -1.0;
		hidden_size = -1;
		input_forget = 0;
		X=NULL;
		W=NULL;
		R=NULL;
		B=NULL;
		sequence_lens=NULL;
		initial_h=NULL;
		initial_c=NULL;
		P=NULL;
		Y=NULL;
		Y_h=NULL;
		Y_c=NULL;
	}

	// inputs
	const Tensor *X;
	const Tensor *W;
	const Tensor *R;
	// optional inputs
	const Tensor *B;
	const Tensor *sequence_lens;
	const Tensor *initial_h;
	const Tensor *initial_c;
	const Tensor *P;
	// optional outputs
	Tensor *Y;
	Tensor *Y_h;
	Tensor *Y_c;

	// Attributes
	std::vector<float> activation_alpha;
	std::vector<float> activation_beta;
	std::vector<std::string> activations; // in order, activations f, g, & h
	float clip;  // negative for no clip
	std::string direction;
	int hidden_size;
	int input_forget;


	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		X->print_tensor(dst, !decorate, !decorate?"":"X");
		dst << ", ";
		W->print_tensor(dst, !decorate, !decorate?"":"W");
		dst << ", ";
		R->print_tensor(dst, !decorate, !decorate?"":"R");
		if( B ) {
			dst << ", ";
			B->print_tensor(dst, !decorate, !decorate?"":"B");
		}
		if( sequence_lens ) {
			dst << ", ";
			sequence_lens->print_tensor(dst, !decorate, !decorate?"":"sequence_lens");
		}
		if( initial_h ) {
			dst << ", ";
			initial_h->print_tensor(dst, !decorate, !decorate?"":"Y_h");
		}
		if( initial_c ) {
			dst << ", ";
			initial_c->print_tensor(dst, !decorate, !decorate?"":"Y_c");
		}
		if( P ) {
			dst << ", ";
			P->print_tensor(dst, !decorate, !decorate?"":"P");
		}
		if( Y->name != "" ) {
			dst << ", ";
			Y->print_tensor(dst, !decorate, !decorate?"":"Y");
		}
		if( Y_h->name != "" && Y_h->isAliasOf==NULL ) {
			dst << ", ";
			Y_h->print_tensor(dst, !decorate, !decorate?"":"Y_h");
		}
		if( Y_c->name != "" && Y_c->isAliasOf==NULL) {
			dst << ", ";
			Y_c->print_tensor(dst, !decorate, !decorate?"":"Y_c");
		}
	}

	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto a : node.attribute() ) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;

			if( a.name() == "activation_alpha" )
				activation_alpha = parse_attribute_floats(a);
			else if( a.name() == "activation_beta" )
				activation_beta = parse_attribute_floats(a);
			else if( a.name() == "activations" )
				activations = parse_attribute_strings(a);
			else if( a.name() == "clip" )
				clip = parse_attribute_float(a);
			else if( a.name() == "direction" )
				direction = parse_attribute_string(a);
			else if( a.name() == "hidden_size" )
				hidden_size = parse_attribute_int(a);
			else if( a.name() == "input_forget" )
				input_forget = parse_attribute_int(a);
			else
				ERROR("Bad attribute " << a.name() << " for LSTM");
		}
	}

	float get_activation_alpha( const std::string &a)
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
	float get_activation_beta( const std::string &a)
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

	virtual void print(std::ostream &dst) const override
	{
		dst << "\t/* LSTM " << std::endl;
		dst << "\t * inputs: " << std::endl;
		dst << "\t *   X = " << X->cname() << std::endl;
		dst << "\t *   W = " << W->cname() << std::endl;
		dst << "\t *   R = " << R->cname() << std::endl;
		dst << "\t *   B = " << (B?B->cname():"") << std::endl;
		dst << "\t *   sequence_lens = " << (sequence_lens?sequence_lens->cname():"") << std::endl;
		dst << "\t *   initial_h = " << (initial_h?initial_h->cname():"") << std::endl;
		dst << "\t *   initial_c = " << (initial_c?initial_c->cname():"") << std::endl;
		dst << "\t *   P = " << (P?P->cname():"") << std::endl;
		dst << "\t * outputs: " << std::endl;
		dst << "\t *   Y = " << Y->cname() << std::endl;
		dst << "\t *   Y_h = " << Y_h->cname() << std::endl;
		dst << "\t *   Y_c = " << Y_c->cname() << std::endl;
		dst << "\t * attributes:" << std::endl;
		dst << "\t * (TBD):" << std::endl;
		dst << "\t */" << std::endl;

		/*
		float (*Y_c)[3] = tensor_node_anonymous_LSTM_0_recursive_2[0];
		float (*ht)[3] = tensor_Y[0];
		float (*X)[2] = tensor_X[0];
		float (*W)[2] = tensor_W[0];
		float (*R)[3] = tensor_R[0];
		*/
		const std::string data_type = X->data_type_str();

		int hs = R->data_dim[2]; //hidden size
		int ds = X->data_dim[2]; //input (data) size
		int bs = X->data_dim[1]; // batch size

		if( X->data_dim[0] != 1 )
			ERROR("Unimplemented: sequence lenght of not 1");


		dst << "\t" << "int hs = " << hs << ";" << std::endl;
		dst << "\t" << "int ds = " << ds << ";" << std::endl;
		dst << "\t" << "int bs = " << bs << ";" << std::endl;
		// index into W, R to get the start of the gate indices
		dst << "\t" << "int iidx = 0;" << std::endl;
		dst << "\t" << "int oidx = hs;" << std::endl;
		dst << "\t" << "int fidx = 2*hs;" << std::endl;
		dst << "\t" << "int cidx = 3*hs;" << std::endl;

		dst << "\t" << "/* Forget gate */" << std::endl;
		dst << "\t" << data_type << " ft[bs][hs];" << std::endl;
		dst << "\t" << "for( int i=0; i<bs; i++)" << std::endl;
		dst << "\t" << "for( int j=0; j<hs; j++) {" << std::endl;
		dst << "\t\t" << "ft[i][j]=0;" << std::endl;
		// Xt*W
		dst << "\t\t" << "for( int k=0; k<ds; k++)" << std::endl;
		dst << "\t\t\t" << "ft[i][j] += X[0][i][k]*W[0][fidx+j][k];" << std::endl;
		// Ht-1*R
		dst << "\t\t" << "for( int k=0; k<hs; k++)" << std::endl;
		dst << "\t\t\t" << "ft[i][j] += Y_h[0][i][k]*R[0][fidx+j][k];" << std::endl;
		if( B ) // Bias
		dst << "\t\t" << "ft[i][j] += B[0][fidx+j];" << std::endl;
		if( P ) // Peephole
		dst << "\t\t" << "ft[i][j] += P[0][fidx+j]*Y_c[0][i][j];" << std::endl;
		// TODO: this is sigmoid. Don't hard-code - activation is an node attribute
		dst << "\t\t" << "ft[i][j] = 1.0f/(1+expf(-ft[i][j]));" << std::endl;
		dst << "\t" << "}" << std::endl;


		dst << "\t" << "/* Input gate */" << std::endl;
		dst << "\t" << data_type << " it[bs][hs];" << std::endl;
		dst << "\t" << "for( int i=0; i<bs; i++)" << std::endl;
		dst << "\t" << "for( int j=0; j<hs; j++) {" << std::endl;
		dst << "\t\t" << "it[i][j]=0;" << std::endl;
		// Xt*W
		dst << "\t\t" << "for( int k=0; k<ds; k++)" << std::endl;
		dst << "\t\t\t" << "it[i][j] += X[0][i][k]*W[0][iidx+j][k];" << std::endl;
		// Ht-1*R
		dst << "\t\t" << "for( int k=0; k<hs; k++)" << std::endl;
		dst << "\t\t\t" << "it[i][j] += Y_h[0][i][k]*R[0][iidx+j][k];" << std::endl;
		if( B ) // Bias
		dst << "\t\t" << "it[i][j] += B[0][iidx+j];" << std::endl;
		if( P ) // Peephole
		dst << "\t\t" << "it[i][j] += P[0][iidx+j]*Y_c[0][i][j];" << std::endl;
		// TODO: this is sigmoid. Don't hard-code - activation is an node attribute
		dst << "\t\t" << "it[i][j] = 1.0f/(1+expf(-it[i][j]));" << std::endl;
		dst << "\t" << "}" << std::endl;


		dst << "\t" << "/* Cell gate */" << std::endl;
		dst << "\t" << data_type << " ct[bs][hs];" << std::endl;
		dst << "\t" << "for( int i=0; i<bs; i++)" << std::endl;
		dst << "\t" << "for( int j=0; j<hs; j++) {" << std::endl;
		dst << "\t\t" << "ct[i][j]=0;" << std::endl;
		// Xt*W
		dst << "\t\t" << "for( int k=0; k<ds; k++)" << std::endl;
		dst << "\t\t\t" << "ct[i][j] += X[0][i][k]*W[0][cidx+j][k];" << std::endl;
		// Ht-1*R
		dst << "\t\t" << "for( int k=0; k<hs; k++)" << std::endl;
		dst << "\t\t\t" << "ct[i][j] += Y_h[0][i][k]*R[0][cidx+j][k];" << std::endl;
		if( B ) // Bias
		dst << "\t\t" << "ct[i][j] += B[0][cidx+j];" << std::endl;
		// TODO: this is tahnf. Don't hard-code - activation is an node attribute
		dst << "\t\t" << "ct[i][j] = tanhf(ct[i][j]);" << std::endl;
		dst << "\t" << "}" << std::endl;


		dst << "\t" << "/* Cell state */" << std::endl;
		dst << "\t" << "for( int i=0; i<bs; i++)" << std::endl;
		dst << "\t" << "for( int j=0; j<hs; j++)" << std::endl;
		dst << "\t\t" << "Y_c[0][i][j] = Y_c[0][i][j]*ft[i][j] + it[i][j]*ct[i][j];" << std::endl;

		dst << "\t" << "/* Output gate */" << std::endl;
		dst << "\t" << data_type << " ot[bs][hs];" << std::endl;
		dst << "\t" << "for( int i=0; i<bs; i++)" << std::endl;
		dst << "\t" << "for( int j=0; j<hs; j++) {" << std::endl;
		dst << "\t\t" << "ot[i][j]=0;" << std::endl;
		// X*W
		dst << "\t\t" << "for( int k=0; k<ds; k++)" << std::endl;
		dst << "\t\t\t" << "ot[i][j] += X[0][i][k]*W[0][oidx+j][k];" << std::endl;
		// Ht-1*R
		dst << "\t\t" << "for( int k=0; k<hs; k++)" << std::endl;
		dst << "\t\t\t" << "ot[i][j] += Y_h[0][i][k]*R[0][oidx+j][k];" << std::endl;
		if( B ) // Bias
		dst << "\t\t" << "ot[i][j] += B[0][oidx+j];" << std::endl;
		if( P ) // Peephole
		dst << "\t\t" << "ot[i][j] += P[0][oidx+j]*Y_c[0][i][j];" << std::endl;
		// TODO: this is tahnf. Don't hard-code - activation is an node attribute
		dst << "\t\t" << "ot[i][j] = 1.0f/(1+expf(-ot[i][j]));" << std::endl;
		dst << "\t" << "}" << std::endl;


		dst << "\t" << "/* Hidden state */" << std::endl;
		dst << "\t" << "for( int i=0; i<bs; i++)" << std::endl;
		dst << "\t" << "for( int j=0; j<hs; j++)" << std::endl;
		// TODO: don't hard-code tanh
		dst << "\t\t" << "Y_h[0][i][j] = ot[i][j] * tanhf(Y_c[0][i][j]);" << std::endl;

	}

	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		if( inputs.size() < 3 || inputs.size() > 8 )
			ERROR("wrong number of inputs to LSTM");


		// Set attribute default values for those attributes that are not set in the model
		if( activations.size() == 0 ) {
			activations.push_back("Sigmoid");
			activations.push_back("Tanh");
			activations.push_back("Tanh");
		}
		if( activations.size() == 6 )
			ERROR("Unimplemented - bidirectional LSTM");
		if( activations.size() != 3 )
			ERROR("Error - bad number of activations attributes");

		if( activation_alpha.size() == 0 ) {
			for( auto &a : activations ) {
				activation_alpha.push_back(get_activation_alpha(a));
			}
		}
		if( activation_alpha.size() != 3 )
			ERROR("Unimplemented/error: not 3 activation alphas");
		if( activation_beta.size() == 0 ) {
			for( auto &a : activations ) {
				activation_beta.push_back(get_activation_beta(a));
			}
		}
		if( activation_beta.size() != 3 )
			ERROR("Unimplemented/error: not 3 activation beta");

		if( direction == "" )
			direction = "forward";
		else
			ERROR("Unimplmeneted: backward and bidirectional LSTM");

		if( hidden_size < 0 )
			ERROR("Must provide hidden_size attribute!");

		X = inputs[0];
		W = inputs[1];
		R = inputs[2];
		//optional inputs. Trailing unprovided inputs can just be left out
		//but non-trailing, unprovided inputs MUST have an empty string as name
		// (guess that means tensors MAY NOT have an empty string as name?)
		if( inputs.size() > 3 && inputs[3]->name != "" )
			B = inputs[3];
		if( inputs.size() > 4 && inputs[4]->name != "" )
			sequence_lens = inputs[4];
		if( inputs.size() > 5 && inputs[5]->name != "" )
			initial_h = inputs[5];
		if( inputs.size() > 6 && inputs[6]->name != "" )
			initial_c = inputs[6];
		if( inputs.size() > 7 && inputs[7]->name != "" )
			P = inputs[7];


		int seq_length = X->data_dim[0];
		int batch_size = X->data_dim[1];
		//int input_size = X->data_dim[2];
		int num_directions = W->data_dim[0];

		if( num_directions != 1 )
			ERROR("Unimplmeneted: bidirectional LSTM");

		// TODO: write all sorts of assertions here. Or just assume
		// the onnx model is according to specifications?

		Y = new Tensor;
		Y->data_type = X->data_type;
		std::vector<int> y_size({ seq_length, num_directions, batch_size, hidden_size });
		Y->data_dim = y_size;


		// Y_h and Y_c are special: optional as outputs to the rest of the network,
		// but mandatory as outputs to this node itself. Also, they alias
		// on top of initial_[h,c], which is implemented copying initializers
		// for Y_[h,c].
		Y_h = new Tensor;
		Y_h->data_type = X->data_type;
		std::vector<int> yh_size({ num_directions, batch_size, hidden_size });
		Y_h->data_dim = yh_size;

		Y_h->isRecursive=true;
		if( initial_h ) {
			Y_h->isAliasOf = initial_h;
			Y_h->generate=false;
		}
		// TODO: handle the else case: set zero-initializer to Y_h

		Y_c = new Tensor;
		Y_c->data_type = X->data_type;
		std::vector<int> yc_size({ num_directions, batch_size, hidden_size });
		Y_c->data_dim = yc_size;

		Y_c->isRecursive=true;
		if( initial_c ) {
			Y_c->isAliasOf = initial_c;
			Y_c->generate=false;
		}
		// TODO: handle the else case: zero initializer to Y_c


		outputs.push_back(Y);
		outputs.push_back(Y_h);
		outputs.push_back(Y_c);
	}
};
}

