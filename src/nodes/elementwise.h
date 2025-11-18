/* This file is part of onnx2c.
 *
 * Generic Elementwise applied functions node.
 * Some nodes are identical in description, differeing only in
 * the function applied
 * Calculates elementwise Y = func ( A )
 */
namespace toC {

class Elementwise : public Node {
	float alpha, beta, bias, gamma, lambd;

	public:
	Elementwise(std::string op) {
		op_name = op;
		alpha=beta=gamma=bias=0;
		lambd=0.5;

		// TODO: use the double precision version of the arithmetics for
		// double precision input. OTOH - who uses doubles on MCUs?
		if( op == "Abs" )
			operation = [this](const std::string& x){ return math_func("fabs") + "("+x+");"; };
		else if( op == "Acos" )
			operation = [this](const std::string& x){ return math_func("acos") + "("+x+");"; };
		else if( op == "Acosh" )
			operation = [this](const std::string& x){ return math_func("acosh") + "("+x+");"; };
		else if( op == "Asin" )
			operation = [this](const std::string& x){ return math_func("asin") + "("+x+");"; };
		else if( op == "Asinh" )
			operation = [this](const std::string& x){ return math_func("asinh") + "("+x+");"; };
		else if( op == "Atan" )
			operation = [this](const std::string& x){ return math_func("atan") + "("+x+");"; };
		else if( op == "Atanh" )
			operation = [this](const std::string& x){ return math_func("atanh") + "("+x+");"; };
		else if( op == "Ceil" )
			operation = [this](const std::string& x){ return math_func("ceil") + "("+x+");"; };
		else if( op == "Celu" ) {
			alpha=1.0;
			operation = [this](const std::string& x){
				std::string a = std::to_string(alpha);
				return math_func("fmax")+"(0,"+x+") + "+math_func("fmin")+"(0,"+a+"*("+math_func("exp")+"("+x+"/"+a+")-1));"; };
		}
		else if( op == "Cos" )
			operation = [this](const std::string& x){ return math_func("cos") + "("+x+");"; };
		else if( op == "Cosh" )
			operation = [this](const std::string& x){ return math_func("cosh") + "("+x+");"; };
		else if( op == "Floor" )
			operation = [this](const std::string& x){ return math_func("floor") + "("+x+");"; };
		else if( op == "Elu" ) {
			alpha=1.0;
			operation = [this](const std::string& x){
				std::string a = std::to_string(alpha);
				return x+">0 ? "+x+": "+a+"*("+math_func("exp")+"("+x+")-1);"; };
		}
		else if( op == "Erf" )
			operation = [this](const std::string& x){ return math_func("erf") + "("+x+");"; };
		else if( op == "Exp" )
			operation = [this](const std::string& x){ return math_func("exp") + "("+x+");"; };
		else if( op == "HardSigmoid" ) {
			alpha=0.2;
			beta=0.5;
			operation = [this](const std::string& x){
				std::string a = std::to_string(alpha);
				std::string b = std::to_string(beta);
				return math_func("fmax")+"(0, "+math_func("fmin")+"(1, "+a+"*"+x+"+"+b+"));";};
		}
		else if( op == "HardSwish" ) {
			// NB: HardSwish has fixed attributes. parseAttributes() can override
			// these values, but that would be non-conformant ONNX input.
			// I'm betting version 15 of ONNX operands introduce attributes to HardSwish...
			alpha=1.0f/6;
			beta=0.5;
			operation = [this](const std::string& x){
				std::string a = std::to_string(alpha);
				std::string b = std::to_string(beta);
				return x+"*"+math_func("fmax")+"(0, "+math_func("fmin")+"(1, "+a+"*"+x+"+"+b+"));";};
		}
		else if( op == "LeakyRelu" ) {
			alpha=0.01f;
			operation = [this](const std::string& x){
				std::string a = std::to_string(alpha);
				return x+">0 ? "+x+" : " +x+ "*" +a+ ";"; };
		}
		else if( op == "Log" )
			operation = [this](const std::string& x){ return math_func("log") + "("+x+");"; };
		else if( op == "Neg" )
			operation = [](const std::string& x){ return  " -"+x+";"; };
		else if( op == "Not" )
			operation = [](const std::string& x){ return  "!"+x+";"; };
		else if( op == "Reciprocal" )
			// TODO: check ONNX rules for handling division by zero
			operation = [](const std::string& x){ return  "1/"+x+";"; };
		else if( op == "Round" ) {
			// NB: this is incorrect.
			// ONNX specifies "round towards nearest EVEN integer" (i.e. both
			// 1.5 and 2.5 round to 2.0 !?!).
			// If this breaks your model, and you find this comment when debugging,
			// please file an improvement suggestion to ONNX. Or fix this with
			// something clean that does what ONNX wants.
			LOG(WARNING) << "Round operand implementation is not strictly conformant" << std::endl;
			operation = [this](const std::string& x){ return math_func("round") + "("+x+");"; };
		}
		else if( op == "Selu" ) {
			alpha=1.67326319217681884765625f;
			gamma=1.05070102214813232421875f;
			operation = [this](const std::string& x){
				std::string a = std::to_string(alpha);
				std::string c = std::to_string(gamma);
				//`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
				return x+">0 ? "+c+"*"+x+": "+c+"*("+a+"*"+math_func("exp")+"("+x+")-"+a+");"; };
		}
		else if( op == "Shrink" ) {
			operation = [this](const std::string& x){
				std::string b = std::to_string(bias);
				std::string l = std::to_string(lambd);
				// if( x < -l ) y=x+bias; else if ( x > l ) y=x-bias; else y=0;
				std::string rv = x+ " < - "+l+" ? ";  // if( x<-l )
				rv += x + "+" + b;                    //     x+bias
				rv += " : (" + x + " > " +l + " ? " ; // else if ( x>l )
				rv += x + "-" + b;                    //     x-bias
				rv += " : 0);";                       // else 0
				return rv;
			};
		}
		else if( op == "Sigmoid" )
			operation = [this](const std::string& x){ return  "1/(1+"+math_func("exp")+"(-"+x+"));"; };
		else if( op == "Sign" )
			operation = [](const std::string& x){ return  ""+x+"<0?-1:"+x+">0?1:0;"; };
		else if( op == "Sin" )
			operation = [this](const std::string& x){ return math_func("sin") + "("+x+");"; };
		else if( op == "Sinh" )
			operation = [this](const std::string& x){ return math_func("sinh") + "("+x+");"; };
		else if( op == "Softplus" )
			operation = [this](const std::string& x){ return math_func("log") + "("+math_func("exp")+"("+x+")+1);"; };
		else if( op == "Softsign" )
			operation = [this](const std::string& x){ return  ""+x+"/(1+"+math_func("fabs")+"("+x+"));"; };
		else if( op == "Sqrt" )
			operation = [this](const std::string& x){ return math_func("sqrt") + "("+x+");"; };
		else if( op == "Tan" )
			operation = [this](const std::string& x){ return math_func("tan") + "("+x+");"; };
		else if( op == "Tanh" )
			operation = [this](const std::string& x){ return math_func("tanh") + "("+x+");"; };
		else if( op == "ThresholdedRelu" ) {
			alpha=1.0f;
			operation = [this](const std::string& x){
				std::string a = std::to_string(alpha);
				return x+">"+a+" ? "+x+" : 0;"; };
		}
		else
			ERROR("Elementwise operand not implemented: " + op);
	}

	// Each instance of this class should override this lambda with the operation of the node type.
	std::function<const std::string (const std::string & Xidx)> operation =
		[](const std::string& x){ ERROR("onnx2c internal error"); return ""; };


	// NB: not all ONNX operators implemented with Elementwise have attributes.
	// This gets the attributes over an union of all implemented operators
	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
			if( a.name() == "alpha" )
				alpha = parse_attribute_float(a);
			else if( a.name() == "beta" )
				beta = parse_attribute_float(a);
			else if( a.name() == "bias")
				bias = parse_attribute_float(a);
			else if( a.name() == "gamma" )
				gamma = parse_attribute_float(a);
			else if( a.name() == "lambd") // sic - lambda? In the Shrink operator
				lambd = parse_attribute_float(a);

			else
				ERROR("unknown attribute");
		}
	}


	virtual void print(std::ostream &dst) const override
	{
		const Tensor *X = get_input_tensor(0);
		const Tensor *Y = get_output_tensor(0);

		INDT_1 << "/* " << op_name << std::endl;
		INDT_1 << "   Implemented with Elementwise template." << std::endl;
		INDT_1 << "   alpha = " << alpha << std::endl;
		INDT_1 << "   beta = " << beta << std::endl;
		INDT_1 << "*/" << std::endl;

		// print out the loops over all C dimensions.
		// at the same time, create the indexing strings into X and Y
		std::string Xidx = X->is_scalar() ? "*X" : "X";
		std::string Yidx = Y->is_scalar() ? "*Y" : "Y";
		for( unsigned r=0; r< Y->rank(); r++) {
			std::string lv = "i" + std::to_string(r);
			INDT_1 << "for (size_t " << lv << "=0; " << lv << "<" << Y->data_dim[r] << "; " << lv << "++) {" << std::endl;

			Xidx += "[" + lv + "]";
			Yidx += "[" + lv + "]";
		}

		INDT_2 << Yidx << " = " << operation(Xidx) << std::endl;

		for( unsigned r=0; r<Y->rank(); r++) {
			INDT_1 << "}" << std::endl;
		}
	}


	virtual void resolve(void) override
	{
		const Tensor *X = get_input_tensor(0);
		name_input(0, "X");

		Tensor *t = new Tensor;
		t->data_dim = X->data_dim;
		t->data_type = X->data_type;
		register_output(t, "Y");

		set_math_type(X->data_type);
	}
};
}

