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
			operation = [](const std::string& x){ return  "fabs("+x+");"; };
		else if( op == "Acos" )
			operation = [](const std::string& x){ return  "acosf("+x+");"; };
		else if( op == "Acosh" )
			operation = [](const std::string& x){ return  "acoshf("+x+");"; };
		else if( op == "Asin" )
			operation = [](const std::string& x){ return  "asinf("+x+");"; };
		else if( op == "Asinh" )
			operation = [](const std::string& x){ return  "asinhf("+x+");"; };
		else if( op == "Atan" )
			operation = [](const std::string& x){ return  "atanf("+x+");"; };
		else if( op == "Atanh" )
			operation = [](const std::string& x){ return  "atanhf("+x+");"; };
		else if( op == "Ceil" )
			operation = [](const std::string& x){ return  "ceilf("+x+");"; };
		else if( op == "Celu" ) {
			alpha=1.0;
			operation = [this](const std::string& x){
				std::string a = std::to_string(alpha);
				return  "fmax(0,"+x+") + fmin(0,"+a+"*(exp("+x+"/"+a+")-1));"; };
		}
		else if( op == "Cos" )
			operation = [](const std::string& x){ return  "cosf("+x+");"; };
		else if( op == "Cosh" )
			operation = [](const std::string& x){ return  "coshf("+x+");"; };
		else if( op == "Floor" )
			operation = [](const std::string& x){ return  "floorf("+x+");"; };
		else if( op == "Elu" ) {
			alpha=1.0;
			operation = [this](const std::string& x){
				std::string a = std::to_string(alpha);
				return x+">0 ? "+x+": "+a+"*(exp("+x+")-1);"; };
		}
		else if( op == "Erf" )
			operation = [](const std::string& x){ return  "erff("+x+");"; };
		else if( op == "Exp" )
			operation = [](const std::string& x){ return  "expf("+x+");"; };
		else if( op == "HardSigmoid" ) {
			alpha=0.2;
			beta=0.5;
			operation = [this](const std::string& x){
				std::string a = std::to_string(alpha);
				std::string b = std::to_string(beta);
				return "fmax(0, fmin(1, "+a+"*"+x+"+"+b+"));";};
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
				return x+"*fmax(0, fmin(1, "+a+"*"+x+"+"+b+"));";};
		}
		else if( op == "LeakyRelu" ) {
			alpha=0.01f;
			operation = [this](const std::string& x){
				std::string a = std::to_string(alpha);
				return x+">0 ? "+x+" : " +x+ "*" +a+ ";"; };
		}
		else if( op == "Log" )
			operation = [](const std::string& x){ return  "logf("+x+");"; };
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
			operation = [](const std::string& x){ return  "roundf("+x+");"; };
		}
		else if( op == "Selu" ) {
			alpha=1.67326319217681884765625f;
			gamma=1.05070102214813232421875f;
			operation = [this](const std::string& x){
				std::string a = std::to_string(alpha);
				std::string c = std::to_string(gamma);
				//`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
				return x+">0 ? "+c+"*"+x+": "+c+"*("+a+"*exp("+x+")-"+a+");"; };
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
			operation = [](const std::string& x){ return  "1/(1+exp(-"+x+"));"; };
		else if( op == "Sign" )
			operation = [](const std::string& x){ return  ""+x+"<0?-1:"+x+">0?1:0;"; };
		else if( op == "Sin" )
			operation = [](const std::string& x){ return  "sinf("+x+");"; };
		else if( op == "Sinh" )
			operation = [](const std::string& x){ return  "sinhf("+x+");"; };
		else if( op == "Softplus" )
			operation = [](const std::string& x){ return  "logf(exp("+x+")+1);"; };
		else if( op == "Softsign" )
			operation = [](const std::string& x){ return  ""+x+"/(1+fabsf("+x+"));"; };
		else if( op == "Sqrt" )
			operation = [](const std::string& x){ return  "sqrtf("+x+");"; };
		else if( op == "Tan" )
			operation = [](const std::string& x){ return  "tanf("+x+");"; };
		else if( op == "Tanh" )
			operation = [](const std::string& x){ return  "tanhf("+x+");"; };
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
		const Tensor *Y = get_output_tensor(0);
		INDT_1 << "/* " << op_name << std::endl;
		INDT_1 << "   Implemented with Elementwise template." << std::endl;
		INDT_1 << "   alpha = " << alpha << std::endl;
		INDT_1 << "   beta = " << beta << std::endl;
		INDT_1 << "*/" << std::endl;

		// print out the loops over all C dimensions.
		// at the same time, create the indexing strings into X and Y
		std::string Xidx = "X";
		std::string Yidx = "Y";
		for( unsigned r=0; r< Y->rank(); r++) {
			std::string lv = "i" + std::to_string(r);
			INDT_1 << "for (unsigned " << lv << "=0; " << lv << "<" << Y->data_dim[r] << "; " << lv << "++) {" << std::endl;

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
	}
};
}

