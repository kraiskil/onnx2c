/* This file is part of onnx2c.
 *
 * LRN node.
 * When implementing a new node, use this template
 * as a starting point.
 * Replace all occurances of LRN in this file.
 * Some representative dummy implementation provided.
 *
 * The functions here are callbacks from the onnx2c
 * framework. See node.h for more documentation.
 */
namespace toC {

class LRN : public Node {
	public:
	LRN() {
		op_name = "LRN";
		X=Y=NULL;
		alpha = 0.0001;
		beta  = 0.75;
		bias  = 1.0;
		size  = -1; // number of channels to sum over. Mandatory
	}
	/* Node attributes */
	float alpha;
	float beta;
	float bias;
	int   size;

	// input and output
	const Tensor *X;
	const Tensor *Y;


	/* Parse attributes, if this node has them. */
	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
			if( a.name() == "alpha" )
				alpha = parse_attribute_float(a);
			else if( a.name() == "beta" )
				beta = parse_attribute_float(a);
			else if( a.name() == "bias" )
				bias = parse_attribute_float(a);
			else if( a.name() == "size" )
				size = parse_attribute_int(a);
		}
	}


	/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		X = inputs[0];

		if( size == -1 )
			ERROR("LRN: attribute 'size' was not given");

		/* Create output tensors.
		 * Set data dimensions and data type for the created tensors. */
		Tensor *t = new Tensor;
		t->data_dim = X->data_dim;
		t->data_type = X->data_type;
		/* Store the created tensor both as reference in this node, and into
		 * the return value vector! */
		Y = t;
		outputs.push_back(t);

		/* TODO: optional outputs? */
	}


	/* Print the function parameters - use the order they are introduced in the
	 * ONNX documentation */
	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		X->print_tensor_as_const(dst, !decorate);

		dst << ", ";
		Y->print_tensor(dst, !decorate);
	}


	/* Body of the node implementing function */
	virtual void print(std::ostream &dst) const override
	{

		INDT_1 << "/* LRN */" << std::endl;
		INDT_1 << "/* attributes:" << std::endl;
		INDT_1 << "   alpha: " << alpha << std::endl;
		INDT_1 << "   beta:  " << beta << std::endl;
		INDT_1 << "   bias:  " << bias << std::endl;
		INDT_1 << "   size:  " << size << std::endl;
		INDT_1 << "*/" << std::endl;

		INDT_1 << "int N = " << X->data_dim[0] << ";" << std::endl;
		INDT_1 << "int C = " << X->data_dim[1] << ";" << std::endl;
		INDT_1 << "float alpha = " << alpha << ";" << std::endl;
		INDT_1 << "float beta = " << beta << ";" << std::endl;
		INDT_1 << "float bias = " << bias << ";" << std::endl;
		INDT_1 << "int size = " << size << ";" << std::endl;

		// loop over batches and channels
		INDT_1 << "for (unsigned n=0; n<N; n++) {" << std::endl;
		INDT_1 << "for (unsigned c=0; c<C; c++) {" << std::endl;

		// loop over the data channels
		std::string y_idx="";
		for( unsigned r=2; r< X->rank(); r++) {
			std::string lv = "i" + std::to_string(r);
			INDT_1 << "for (unsigned " << lv << "=0; ";
				dst << lv << "<" << X->data_dim[r] << "; ";
				dst << lv << "++) {" << std::endl;

			y_idx += "[" + lv + "]";
		}


			// Calculate square_sum
			INDT_2 << "int start_i = MAX(0,   c-floor((size-1)/2));" << std::endl;
			INDT_2 << "int end_i   = MIN(C-1, c+ceil((size-1)/2));" << std::endl;
			INDT_2 << "float square_sum = 0;" << std::endl;
			INDT_2 << "for (unsigned i=start_i; i<=end_i; i++) {" << std::endl;
				INDT_3 << "square_sum += pow(" << X->cname() <<"[n][i]"<<y_idx <<", 2);" << std::endl;
			INDT_2 << "}" << std::endl;

			INDT_2 << Y->cname() << "[n][c]" << y_idx << "=" << X->cname() <<"[n][c]" << y_idx << "/" << std::endl;
			/// (bias + alpha / size * square_sum[n, c, d1, ..., dk] ) ^ beta
			INDT_2 <<     "pow(bias + alpha/size * square_sum, beta);" << std::endl;


		// close loop over all dimensions
		for( unsigned r=0; r< X->rank(); r++)
			INDT_1 << "}" << std::endl;
	}
};
}

