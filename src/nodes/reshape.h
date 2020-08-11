
namespace toC {

class Reshape : public Node {
	public:
	Reshape() {
		op_name = "Reshape";
	}

	virtual void print(std::ostream &dst) const
	{
		if( inputs.size() != 2 )
			ERROR("wrong number of inputs to Reshape");
		if( outputs.size() != 1 )
			ERROR("wrong number of outputs from Reshape");
		std::string type = inputs[0]->data_type_str();

		/* TODO: is there ANY case where a reshape needs to re-order the internal data layout ? */
		/* TODO: and if not - check that at least gcc can get rid of this copy! (So onnx2c doesn't need to) */
		dst << "\t/*Reshape*/" << std::endl;
		dst << "\t" << type << " *data = (" << type << "*)" << inputs[0]->cname() << ";" << std::endl;
		dst << "\t" << type << " *reshaped = (" << type << "*)" << outputs[0]->cname() << ";" << std::endl;

		dst << "\t" << "for( uint32_t i=0; i<" << inputs[0]->data_num_elem << "; i++ )" << std::endl;
		dst << "\t\treshaped[i] = data[i];" << std::endl;
		dst << std::endl;
	}
 
	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs)
	{
		const Tensor *data = inputs[0];
		const Tensor *shape = inputs[1];
		if( typeConstraint_int64(shape) == false )
			ERROR("Incorrect input for node"); 


		if( shape->initialize == false ) {
			ERROR("Reshaping to a run-time defined shape is not supported");
		}

		int64_t *new_shape = (int64_t*)(shape->data_buffer);

		/* TODO: uint64_t does not overflow here, right?
		 * Because if it does, you probably aren't targetting edge computing? */
		uint64_t shape_num_elem=1;
		for(int i=0; i<shape->data_num_elem; i++) {
			// dimension=0: unchanged from input
			// dimension=-1: infer from dim
			if( new_shape[i] <= 0 )
				ERROR("Unimplemented: reshaping to shapes 0 or -1");

			shape_num_elem *= new_shape[i];
		}

		if( shape_num_elem != (uint64_t)data->data_num_elem )
			ERROR("Reshape: Wrong amount of input for requested shape");

		Tensor *rv = new Tensor;
		// TODO: what if B has more dimensions? Check ONNX semantics
		for( int i=0; i<shape->data_num_elem; i++) {
			int64_t d = new_shape[i];
			rv->data_dim.push_back(d);
		}

		rv->data_type = data->data_type;
		rv->data_num_elem = data->data_num_elem;
		outputs.push_back(rv);
	}
};
}
