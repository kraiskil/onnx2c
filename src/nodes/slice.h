/* This file is part of onnx2c.
 *
 * Slice node - return a specified part of the input 'data' tensor.
 */
namespace toC {

class Slice : public Node {
	public:
	Slice() {
		op_name = "Slice";
		output=data=starts=ends=axes=steps=NULL;
	}

	// input and output
	const Tensor *output;
	const Tensor *data;
	const Tensor *starts;
	const Tensor *ends;
	// optional inputs
	const Tensor *axes;
	const Tensor *steps;

	// contents of the input tensors (or default values), padded
	// to output dimensions
	std::vector<int>sta;
	std::vector<int>en;
	std::vector<int>ax;
	std::vector<int>stp;

	int index_of_content(const Tensor *t, int c)
	{
		for( int i=0; i<t->data_num_elem(); i++)
		{
			if( t->get_data_element(i) == c )
				return i;
		}
		return -1;
	}

	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		data = inputs[0];
		starts = inputs[1];
		ends = inputs[2];

		if( starts->isConst == false )
			ERROR("Non-const inputs to Slice not handled");
		if( ends->isConst == false )
			ERROR("Non-const inputs to Slice not handled");

		if (inputs.size() > 3)
			axes = inputs[3];
		if (inputs.size() > 4)
			steps = inputs[4];

		// the output tensor
		Tensor *t = new Tensor;


		int ddim = data->data_dim.size();
		sta.resize(ddim);
		en.resize(ddim);
		ax.resize(ddim);
		stp.resize(ddim);

		// Set defaults. Override later if required
		for( unsigned d=0; d<data->rank(); d++) {
			sta[d] = 0;
			en[d]  = data->data_dim[d];
			ax[d]  = d;
			stp[d] = 1;
		}

		// if axes are not provided as input, the rest of the limits must be provided in full
		// or we can't know which axes a limit applies to
		int expected_size;
		if( !axes )
			expected_size=ddim;
		else
			expected_size=axes->data_num_elem();

		if( starts->data_num_elem() != expected_size )
			ERROR("Input 'starts' does not have correct amount of elements");
		if( ends->data_num_elem() != expected_size )
			ERROR("Input 'ends' does not have correct amount of elements");
		if( steps && steps->data_num_elem() != expected_size )
			ERROR("Input 'steps' does not have correct amount of elements");


		// Default values are in place. Override with given values
		if( axes ) {
			for( int i=0; i<axes->data_num_elem(); i++ ) {
				int d = axes->get_data_element(i);
				if( d < 0 )
					d = ddim + d;
				sta[d] = starts->get_data_element(i);
				en[d]  = ends->get_data_element(i);
				if( steps )
					stp[d] = steps->get_data_element(i);
			}
		}
		else {
			for( unsigned d=0; d<data->rank(); d++ ) {
				sta[d] = starts->get_data_element(d);
				en[d]  = ends->get_data_element(d);
				if( steps )
					stp[d] = steps->get_data_element(d);
			}
		}


		// Prune up corner cases: out of range indexing etc. and calculate output
		for( unsigned d=0; d<data->rank(); d++) {
			int s=sta[d];
			int e=en[d];
			int st=stp[d];
			int in_size = data->data_dim[d];

			if( s < 0 )
				s = in_size + s;
			if( e < 0 )
				e = in_size + e;
			if( s>=in_size )
				s=in_size;
			if( e>=in_size )
				e=in_size;

			sta[d]=s;
			en[d]=e;

			// calculate the output dimension
			// ok, there probably exist a closed form for this algorithm.
			// but I'm tired :)
			int num=0;
			if( s>e ) {
				std::swap(s,e);
				// start is inclusive, end exclusive. "shift left"
				s--;
				e--;
				if( s < 0 )
					s=0;
				if( e > in_size )
					e=in_size;
				st=-st;
			}
			for(int n=s; n<e; n+=st)
				num++;
			t->data_dim.push_back(num);
			if( num <= 0 )
				// https://github.com/onnx/onnx/issues/3724
				ERROR("Unimplemented: tensor sliced to have dimension of size 0");
		}

		t->data_type = data->data_type;
		output = t;
		outputs.push_back(t);
	}


	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		data->print_tensor_as_const(dst, !decorate);
		dst << ", ";
		starts->print_tensor_as_const(dst, !decorate);
		dst << ", ";
		ends->print_tensor_as_const(dst, !decorate);

		if (axes) {
			dst << ", ";
			axes->print_tensor_as_const(dst, !decorate);
		}
		if (steps) {
			dst << ", ";
			steps->print_tensor_as_const(dst, !decorate);
		}

		dst << ", ";
		output->print_tensor(dst, !decorate);

	}


	/* Body of the node implementing function */
	virtual void print(std::ostream &dst) const override
	{

		INDT_1 << "/* Slice */" << std::endl;
		std::string out_idx, in_idx;

		// Loop over output dimensions & create the indexing arrays
		for( unsigned d=0; d< output->rank(); d++) {
			int64_t s;  //start
			int64_t e;  //end
			int32_t st; //step
			int in_size = data->data_dim[d];
			s=sta[d];
			e=en[d];
			st=stp[d];

			// start and end have different semantics.
			// start index is inclusive, end exclusive.
			if( s>e && s==in_size)
				s--;


			std::string iv = "i" + std::to_string(d);
			std::string ov = "o" + std::to_string(d);
			INDT_1 << "for (unsigned " << iv << "=" << s << ", " << ov << "=0; ";
			//dst    << iv << "<" << e  << "; ";
			dst    << ov << "<" << output->data_dim[d]  << "; ";
			dst    << iv << "+=" << st <<", " << ov << "++) {" << std::endl;

			out_idx += "[" + ov + "]";
			in_idx  += "[" + iv + "]";
		}

		// Copy over data from input to output
		INDT_2 << output->cname() << out_idx << " = " << data->cname() << in_idx << ";" << std::endl;

		// close loops over output dimensions
		for( unsigned r=0; r<output->rank(); r++) {
			INDT_1 << "}" << std::endl;
		}
	}
};
}

