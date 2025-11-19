/* This file is part of onnx2c.
 *
 * Slice node - return a specified part of the input 'data' tensor.
 *
 * ONNX operator set version 10 changed Slice to take 'axes', 'starts'
 * and 'ends' as tensors instead of attributes. This implementation
 * works with both.
 *
 * Slicing a dimension to size zero is not supported.
 * See https://github.com/onnx/onnx/issues/3724 on a good description
 * of related problems.
 */
namespace toC {

class Slice : public Node {
	public:
	Slice()
	{
		op_name = "Slice";
	}

	// contents of the input tensors, attributes or default values; padded
	// to output dimensions in resolve(void).
	std::vector<int64_t> sta;
	std::vector<int64_t> en;
	std::vector<int64_t> ax;
	std::vector<int64_t> stp;

	virtual void parseAttributes(onnx::NodeProto& node) override
	{
		for (const auto& a : node.attribute()) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
			if (a.name() == "axes")
				ax = parse_attribute_ints(a);
			else if (a.name() == "starts")
				sta = parse_attribute_ints(a);
			else if (a.name() == "ends")
				en = parse_attribute_ints(a);
			else
				ERROR("Unknonw attribute to slice");
		}
	}

	virtual void resolve(void) override
	{
		const Tensor* data = get_input_tensor(0);
		const Tensor* starts = nullptr;
		const Tensor* ends = nullptr;
		const Tensor* axes = nullptr;
		const Tensor* steps = nullptr;
		name_input(0, "data");

		if (get_number_of_inputs() > 1) {
			starts = get_input_tensor(1);
			name_input(1, "starts");
		}
		if (get_number_of_inputs() > 2) {
			ends = get_input_tensor(2);
			name_input(2, "ends");
		}

		if (starts && starts->isConst == false)
			ERROR("Non-const inputs to Slice not handled");
		if (ends && ends->isConst == false)
			ERROR("Non-const inputs to Slice not handled");

		if (get_number_of_inputs() > 3) {
			axes = get_input_tensor(3);
			name_input(3, "axes");
		}
		if (get_number_of_inputs() > 4) {
			steps = get_input_tensor(4);
			name_input(4, "steps");
		}

		// the output tensor
		Tensor* t = new Tensor;

		// Create local working copies of the similarly
		// named class variables.
		// At the end of this function, these working copies
		// are copied back to the "originals".
		int ddim = data->data_dim.size();
		std::vector<int64_t> sta_(ddim);
		std::vector<int64_t> en_(ddim);
		std::vector<int64_t> ax_(ddim);
		std::vector<int64_t> stp_(ddim);

		// ax = [0,1,2,...], if not given.
		if (onnx_ir_version <= 9 && ax.size() == 0)
			for (unsigned d = 0; d < data->rank(); d++)
				ax.push_back(d);

		// Set defaults. Override later if required
		for (unsigned d = 0; d < data->rank(); d++) {
			sta_[d] = 0;
			en_[d] = data->data_dim[d];
			ax_[d] = d;
			stp_[d] = 1;
		}

		// if axes are not provided as input, the rest of the limits must be provided in full
		// or we can't know which axes a limit applies to
		int expected_size; // of starts, ends & steps
		if (!axes)
			expected_size = ddim;
		else if (onnx_ir_version > 9)
			expected_size = axes->data_num_elem();
		else
			expected_size = ax.size();

		if (starts && starts->data_num_elem() != expected_size)
			ERROR("Input 'starts' does not have correct amount of elements");
		if (ends && ends->data_num_elem() != expected_size)
			ERROR("Input 'ends' does not have correct amount of elements");
		if (steps && steps->data_num_elem() != expected_size)
			ERROR("Input 'steps' does not have correct amount of elements");

		// Default values are in place. Override with given values
		if (axes) {
			for (int i = 0; i < axes->data_num_elem(); i++) {
				int d = axes->get_data_element(i);
				if (d < 0)
					d = ddim + d;
				sta_[d] = starts->get_data_element(i);
				en_[d] = ends->get_data_element(i);
				if (steps)
					stp_[d] = steps->get_data_element(i);
			}
		}
		else if (onnx_ir_version > 9) {
			for (unsigned d = 0; d < data->rank(); d++) {
				sta_[d] = starts->get_data_element(d);
				en_[d] = ends->get_data_element(d);
				if (steps)
					stp_[d] = steps->get_data_element(d);
			}
		}
		else {
			for (unsigned i = 0; i < ax.size(); i++) {
				int d = ax[i];
				if (d < 0)
					d = ddim + d;
				sta_[d] = sta[i];
				en_[d] = en[i];
				if (steps)
					stp_[d] = 1;
			}
		}

		// Prune up corner cases: out of range indexing etc. and calculate output
		for (unsigned d = 0; d < data->rank(); d++) {
			int64_t s = sta_[d];
			int64_t e = en_[d];
			int64_t st = stp_[d];
			int64_t in_size = data->data_dim[d];

			if (s < 0)
				s = in_size + s;
			if (e < 0)
				e = in_size + e;
			if (s >= in_size)
				s = in_size;
			if (e >= in_size)
				e = in_size;

			sta_[d] = s;
			en_[d] = e;

			// calculate the output dimension
			// ok, there probably exist a closed form for this algorithm.
			// but I'm tired :)
			int num = 0;
			if (s > e) {
				std::swap(s, e);
				// start is inclusive, end exclusive. "shift left"
				s--;
				e--;
				if (s < 0)
					s = 0;
				if (e > in_size)
					e = in_size;
				st = -st;
			}
			for (int n = s; n < e; n += st)
				num++;
			t->data_dim.push_back(num);
			if (num <= 0)
				// https://github.com/onnx/onnx/issues/3724
				ERROR("Unimplemented: tensor sliced to have dimension of size 0");
		}

		ax = ax_;
		sta = sta_;
		en = en_;
		stp = stp_;

		t->data_type = data->data_type;
		register_output(t, "output");
	}

	/* Body of the node implementing function */
	virtual void print(std::ostream& dst) const override
	{
		const Tensor* data = get_input_tensor(0);
		const Tensor* output = get_output_tensor(0);

		std::string out_idx, in_idx;

		// Loop over output dimensions & create the indexing arrays
		for (unsigned d = 0; d < output->rank(); d++) {
			int64_t s;  //	start
			int64_t e;  //	end
			int32_t st; //	step
			int in_size = data->data_dim[d];
			s = sta[d];
			e = en[d];
			st = stp[d];

			// start and end have different semantics.
			// start index is inclusive, end exclusive.
			if (s > e && s == in_size)
				s--;

			std::string iv = "i" + std::to_string(d);
			std::string ov = "o" + std::to_string(d);
			INDT_1 << "for (unsigned " << iv << "=" << s << ", " << ov << "=0; ";
			//	dst    << iv << "<" << e  << "; ";
			dst << ov << "<" << output->data_dim[d] << "; ";
			dst << iv << "+=" << st << ", " << ov << "++) {" << std::endl;

			out_idx += "[" + ov + "]";
			in_idx += "[" + iv + "]";
		}

		// Copy over data from input to output
		INDT_2 << "output" << out_idx << " = data" << in_idx << ";" << std::endl;

		// close loops over output dimensions
		for (unsigned r = 0; r < output->rank(); r++) {
			INDT_1 << "}" << std::endl;
		}
	}
};
} // namespace toC
