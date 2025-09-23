/* This file is part of onnx2c.
 *
 * Generic node for two input tensors.
 * Calculates elementvise C = A <op> B
 */
namespace toC {

class Elementwise_2 : public Node {
	public:

	// Each instance of this class should override this lambda with the operation of the node type.
	std::function<const std::string (const std::string&, const std::string&)> operation =
		[](const std::string &a, const std::string &b){ ERROR("onnx2c internal error"); return ""; };

	bool output_is_bool;

	// Union of attributes over implemented nodes
	std::string shift_dir;
	int fmod;

	Elementwise_2(std::string op) {
		op_name = op;
		output_is_bool = false;
		fmod=0;
		shift_dir="NOT_GIVEN"; // mandatory for BitShift, but no default

		if( op == "Add" )
			operation = [](const std::string& a, const std::string& b)
				{ return  a+"+"+b+";"; };
		else if( op == "And" )
			operation = [](const std::string& a, const std::string& b)
				{ return  a+"&"+b+";"; };
		else if( op == "BitShift" )
			operation = [this](const std::string& a, const std::string& b)
				{ if( shift_dir=="RIGHT")
					return  a+">>"+b+";";
				  else
					return  a+"<<"+b+";";
				};
		else if (op == "Div" )
			operation = [](const std::string& a, const std::string& b)
				{ return  a+"/"+b+";"; };
		else if (op == "Equal" ) {
			output_is_bool = true;
			// NB: specs don't define what kind of equality is meant when inputs are floating point
			// This passes currently existing ONNX unit tests...
			operation = [](const std::string& a, const std::string& b)
				{ return  a+"=="+b+";"; };
		}
		else if (op == "Greater" ) {
			output_is_bool = true;
			operation = [](const std::string& a, const std::string& b)
				{ return  a+">"+b+";"; };
		}
		else if (op == "GreaterOrEqual" ) {
			output_is_bool = true;
			operation = [](const std::string& a, const std::string& b)
				{ return  a+">="+b+";"; };
		}
		else if (op == "Less" ) {
			output_is_bool = true;
			operation = [](const std::string& a, const std::string& b)
				{ return  a+"<"+b+";"; };
		}
		else if (op == "LessOrEqual" ) {
			output_is_bool = true;
			operation = [](const std::string& a, const std::string& b)
				{ return  a+"<="+b+";"; };
		}
		else if (op == "Mod" )
			operation = [this](const std::string& a, const std::string& b)
				{
					if( fmod )
						return "fmod("+a+","+b+");";
					else
						ERROR("Non fmod Mod operator definition is not clear in ONNX specification");
				};
		else if (op == "Mul" )
			operation = [](const std::string& a, const std::string& b)
				{ return  a+"*"+b+";"; };
		else if (op == "Or" ) {
			output_is_bool = true; // inputs are bool too...
			operation = [](const std::string& a, const std::string& b)
				{ return  a+"||"+b+";"; };
		}
		else if (op == "Pow" )
			// TODO: don't use powf for integers!
			operation = [](const std::string& a, const std::string& b)
				{ return  "powf("+a+","+b+");"; };
		else if (op == "PRelu" )
			operation = [](const std::string& a, const std::string& b)
				{ return  a+"<0?"+a+"*"+b+":"+a+";"; };
		else if (op == "Xor" ) {
			output_is_bool = true; // inputs are bool too...
			operation = [](const std::string& a, const std::string& b)
				{ return  a+"^"+b+";"; };
		}
		else if (op == "Sub" )
			operation = [](const std::string& a, const std::string& b)
				{ return  a+"-"+b+";"; };
		else
			ERROR("Elementwise_2 operand " + op + " not implemented");
	}

	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
			if( a.name() == "direction" )
				shift_dir = parse_attribute_string(a);
			else if( a.name() == "fmod" )
				fmod = parse_attribute_int(a);
			else
				ERROR("unknown attribute");
		}
	}

	virtual void print(std::ostream &dst) const override
	{
		INDT_1 << "/* " << op_name  << std::endl;
		INDT_1 << "   Implemented with Elementwise_2 template." << std::endl;
		INDT_1 << "   Attributes (these are the union of attributes for all 2-element-wise" << std::endl;
		INDT_1 << "               operands. So most likely these values are ignored by onnx2c)." << std::endl;
		INDT_1 << "   shift_dir: " << shift_dir << std::endl;
		INDT_1 << "   fmod: " << fmod << std::endl;
		INDT_1 << " */" << std::endl;

		// C = A ? B
		const Tensor *A = get_input_tensor(0);
		const Tensor *B = get_input_tensor(1);
		const Tensor *C = get_output_tensor(0);

		// if either A or B does not have enough dimensions, prepend
		// dimensions of 1 to match rank of C
		// TODO: explain why. This makes no sense. Can't index into A or B with
		//       more dimensions than they have??
		std::vector<int> padA = A->data_dim;
		std::vector<int> padB = B->data_dim;
		for( unsigned i=0; i< (C->rank() - A->rank()); i++)
			padA.insert(padA.begin(), 0);
		for( unsigned i=0; i< (C->rank() - B->rank()); i++)
			padB.insert(padB.begin(), 0);

		// print out the loops over all C dimensions.
		// at the same time, create the indexing strings into A and B
		std::string Aidx = A->is_scalar() ? "*A" : "A";
		std::string Bidx = B->is_scalar() ? "*B" : "B";
		std::string Cidx = C->is_scalar() ? "*C" : "C";

		for( unsigned r=0; r<C->rank(); r++) {
			std::string lv = "i" + std::to_string(r);
			INDT_1 << "for (unsigned " << lv << "=0; " << lv << "<" << C->data_dim[r] << "; " << lv << "++) {" << std::endl;

			if (!A->is_scalar() ) {
				if (padA[r]==1)
					Aidx += "[0]";
				else if(padA[r]!=0)
					Aidx += "[" + lv + "]";
			}
			if (!B->is_scalar() ) {
				if (padB[r]==1)
					Bidx += "[0]";
				else if(padB[r]!=0)
					Bidx += "[" + lv + "]";
			}
			// TODO: "if C->is_scalar()"?
			// but then again, can the result ever be a scalar?
			Cidx +="[" + lv + "]";
		}


		if( options.quantize ) {
			INDT_2 << "int32_t tmp = " << operation(Aidx, Bidx) << ";" << std::endl;
			// TODO: division amount here depends on operand
			INDT_2 << "tmp = tmp/2;" << std::endl;
			INDT_2 << "tmp = tmp > 127?127:tmp;" << std::endl;
			INDT_2 << "tmp = tmp < -127?-127:tmp;" << std::endl;
			INDT_2 << Cidx << "= tmp;" << std::endl;
		}
		else
			INDT_2 << Cidx << " = " << operation(Aidx, Bidx) << ";" << std::endl;

		for( unsigned r=0; r<C->rank(); r++) {
			INDT_1 << "}" << std::endl;
		}
	}


	virtual void resolve(void) override
	{
		const Tensor *A = get_input_tensor(0);
		const Tensor *B = get_input_tensor(1);
		name_input(0, "A");
		name_input(1, "B");

		std::vector<int> result_dim;
		multidirectional_broadcast_size(A->data_dim, B->data_dim, result_dim);

		Tensor *t = new Tensor;
		t->data_dim = result_dim;
		if( output_is_bool )
			t->data_type = onnx::TensorProto_DataType_BOOL;
		else
			t->data_type = A->data_type;
		register_output(t, "C");
	}
};
}

