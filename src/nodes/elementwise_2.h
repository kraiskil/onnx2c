/* This file is part of onnx2c.
 *
 * Generic node for two input tensors.
 * Calculates elementvise C = A <op> B
 */

namespace toC {

class Elementwise_2 : public Node {
	public:
	// Each instance of this class should override this lambda with the operation of the node type.
	std::function<const std::string(const std::string&, const std::string&)> operation =
	    [](const std::string& a, const std::string& b) { ERROR("onnx2c internal error"); return ""; };

	bool output_is_bool;

	// Union of attributes over implemented nodes
	std::string shift_dir;
	int fmod;

	Elementwise_2(std::string op)
	{
		op_name = op;
		output_is_bool = false;
		fmod = 0;
		shift_dir = "NOT_GIVEN"; // mandatory for BitShift, but no default

		if (op == "Add")
			operation = [](const std::string& a, const std::string& b) { return a + "+" + b + ";"; };
		else if (op == "And")
			operation = [](const std::string& a, const std::string& b) { return a + "&" + b + ";"; };
		else if (op == "BitShift")
			operation = [this](const std::string& a, const std::string& b) {
				if (shift_dir == "RIGHT")
					return a + ">>" + b + ";";
				else
					return a + "<<" + b + ";";
			};
		else if (op == "Div")
			operation = [](const std::string& a, const std::string& b) { return a + "/ " + b + ";"; };
		else if (op == "Equal") {
			output_is_bool = true;
			// NB: specs don't define what kind of equality is meant when inputs are floating point
			// This passes currently existing ONNX unit tests...
			operation = [](const std::string& a, const std::string& b) { return a + "==" + b + ";"; };
		}
		else if (op == "Greater") {
			output_is_bool = true;
			operation = [](const std::string& a, const std::string& b) { return a + ">" + b + ";"; };
		}
		else if (op == "GreaterOrEqual") {
			output_is_bool = true;
			operation = [](const std::string& a, const std::string& b) { return a + ">=" + b + ";"; };
		}
		else if (op == "Less") {
			output_is_bool = true;
			operation = [](const std::string& a, const std::string& b) { return a + "<" + b + ";"; };
		}
		else if (op == "LessOrEqual") {
			output_is_bool = true;
			operation = [](const std::string& a, const std::string& b) { return a + "<=" + b + ";"; };
		}
		else if (op == "Mod")
			operation = [this](const std::string& a, const std::string& b) {
				if (fmod)
					return math_func("fmod") + "(" + a + "," + b + ");";
				else
					ERROR("Non fmod Mod operator definition is not clear in ONNX specification");
			};
		else if (op == "Mul")
			operation = [](const std::string& a, const std::string& b) { return a + "*" + b + ";"; };
		else if (op == "Or") {
			output_is_bool = true; // inputs are bool too...
			operation = [](const std::string& a, const std::string& b) { return a + "||" + b + ";"; };
		}
		else if (op == "Pow")
			// TODO: don't use powf for integers!
			operation = [this](const std::string& a, const std::string& b) { return math_func("pow") + "(" + a + "," + b + ");"; };
		else if (op == "PRelu")
			operation = [](const std::string& a, const std::string& b) { return a + "<0?" + a + "*" + b + ":" + a + ";"; };
		else if (op == "Xor") {
			output_is_bool = true; // inputs are bool too...
			operation = [](const std::string& a, const std::string& b) { return a + "^" + b + ";"; };
		}
		else if (op == "Sub")
			operation = [](const std::string& a, const std::string& b) { return a + "-" + b + ";"; };
		else
			ERROR("Elementwise_2 operand " + op + " not implemented");
	}

	virtual void parseAttributes(onnx::NodeProto& node) override
	{
		for (const auto& a : node.attribute()) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
			if (a.name() == "direction")
				shift_dir = parse_attribute_string(a);
			else if (a.name() == "fmod")
				fmod = parse_attribute_int(a);
			else
				ERROR("unknown attribute");
		}
	}

	virtual void print(std::ostream& dst) const override
	{
		INDT_1 << "/* " << op_name << std::endl;
		INDT_1 << "   Implemented with Elementwise_2 template." << std::endl;
		INDT_1 << "   Attributes (these are the union of attributes for all 2-element-wise" << std::endl;
		INDT_1 << "               operands. So most likely these values are ignored by onnx2c)." << std::endl;
		INDT_1 << "   shift_dir: " << shift_dir << std::endl;
		INDT_1 << "   fmod: " << fmod << std::endl;
		INDT_1 << " */" << std::endl;

		// C = A ? B
		const Tensor* A = get_input_tensor(0);
		const Tensor* B = get_input_tensor(1);
		const Tensor* C = get_output_tensor(0);

		for (unsigned r = 0; r < C->rank(); r++) {
			std::string lv = "i" + std::to_string(r);
			INDT_1 << "for (unsigned " << lv << "=0; " << lv << "<" << C->data_dim[r] << "; " << lv << "++)" << std::endl;
		}

		INDT_1 << "{" << std::endl;

		std::string Aidx = broadcast(A, "A", C->rank());
		std::string Bidx = broadcast(B, "B", C->rank());
		std::string Cidx = broadcast(C, "C", C->rank());

		INDT_2 << Cidx << " = " << operation(Aidx, Bidx) << ";" << std::endl;

		INDT_1 << "}" << std::endl;
	}

	virtual void resolve(void) override
	{
		const Tensor* A = get_input_tensor(0);
		const Tensor* B = get_input_tensor(1);
		name_input(0, "A");
		name_input(1, "B");

		std::vector<int> result_dim;
		multidirectional_broadcast_size(A->data_dim, B->data_dim, result_dim);

		Tensor* t = new Tensor;
		t->data_dim = result_dim;
		if (output_is_bool)
			t->data_type = onnx::TensorProto_DataType_BOOL;
		else
			t->data_type = A->data_type;
		register_output(t, "C");

		set_math_type(A->data_type);
	}
};
} // namespace toC
