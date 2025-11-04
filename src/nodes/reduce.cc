/* This file is part of onnx2c.
 *
 * Reduce Operations
 *
 */
#include "reduce.h"
namespace toC {

void print_loop_closes_over_dims(std::ostream &dst, unsigned rank)
{
    for (unsigned r = 0; r < rank; r++) {
        INDT(rank-r) << "}" << std::endl;
    }
}

int Reduce::get_number_of_reduced_elements(void) const
{
    int reduced_size = 1;
    if (norm_axes.empty()) {
        for (size_t dim: input->data_dim) {
            reduced_size *= dim;
        }
    } else {
        for (size_t axis: norm_axes) {
            reduced_size *= input->data_dim[axis];
        }
    }
    return reduced_size;
}

void Reduce::parseAttributes( onnx::NodeProto &node )
{
    for( const auto& a : node.attribute() ) {
        LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
        if( a.name() == "axes" ) // optinal (might be empty)
            axes = parse_attribute_ints(a);
		else if( a.name() == "keepdims" ) { 
            keepdims = (bool) parse_attribute_int(a);
            keepdims = keepdims != 0 ? 1:0; // default is 1
        }
        else
            ERROR("Ignoring attribute " + a.name() + " for node Reduce/" + onnx_name);
    }
}


std::string Reduce::print_and_return_o_iterator(std::ostream &dst) const
{
    std::string idx = "";
    for (unsigned r = 0; r < output->rank(); r++) {
        std::string lv = "i" + std::to_string(r);
        INDT(r+1) << "for (unsigned " << lv << " = 0; ";
        dst << lv << "<" << output->data_dim[r] << "; ";
        dst << lv << "++) {" << std::endl;

        if (norm_axes.empty()) {
            if (keepdims) idx += "[0]";
        } else {
            idx += "[" + lv + "]";
        }
    }
    if (idx.empty()){
        idx = "[0]";
    }
    return idx;
}

std::pair<std::string, std::string> Reduce::print_and_return_io_iterator(std::ostream &dst) const
{
    std::string inp_idx = "";  // iterates also over input axis dimensions
    std::string out_idx = "";  // will be skipped for the axis dimensions that should be reduced
    
    for (unsigned r = 0; r < input->rank(); r++) {
        std::string lv = "i" + std::to_string(r);
        INDT(r+1) << "for (unsigned " << lv << " = 0; ";
        dst << lv << "<" << input->data_dim[r] << "; ";
        dst << lv << "++) {" << std::endl;

        inp_idx += "[" + lv + "]"; 

        if (norm_axes.empty()) {
            if (keepdims) out_idx += "[0]";
        } else {
            
            if (std::find(norm_axes.begin(), norm_axes.end(), r) != norm_axes.end()) {
                if (keepdims) out_idx += "[0]";
            } else {
                    out_idx += "[" + lv + "]";
                }
            }
    }
   if (out_idx.empty()){
        out_idx = "[0]";
    }
    return std::make_pair(inp_idx, out_idx);
}

std::vector<size_t> Reduce::normalized_axes(const Tensor *t) const
{
    std::vector<size_t> normalized_axes;
    for (int64_t axis: axes) {
        if (axis < 0) {
            axis += t->rank();
        }
        if (axis < 0 || static_cast<size_t>(axis) >= t->data_dim.size()) {
            ERROR("Invalid axis " + std::to_string(axis) + " for input tensor with " 
            + std::to_string(t->data_dim.size()) + " dimensions.");
        }
        normalized_axes.push_back(static_cast<size_t>(axis));
    }
    return normalized_axes;
}

void Reduce::resolve(void)
{
    input = get_input_tensor(0);
	name_input(0, "x");

    if (get_number_of_inputs() >= 2) {
        const Tensor *axes_tensor = get_input_tensor(1);
		name_input(1, "axes");

        if ( !axes_tensor->isConst )
            ERROR("Reducing on run-time defined axes not supported");
        
        assert(axes_tensor->data_type == onnx::TensorProto_DataType_INT64);

        assert(axes.size() == 0);
        for (int i = 0; i < axes_tensor->data_num_elem(); i++) {
            axes.push_back(((int64_t*)axes_tensor->data_buffer)[i]);
        }
    }

    set_math_type( input->data_type );

	Tensor *t = new Tensor;

    Tensor* x = get_input_tensor(0);
    std::string type = x->data_type_str();

    std::string type_min_value;
    std::string type_max_value;
    std::string type_0_value;
    std::string type_1_value;

    if( type == "float" )
    {
        type_min_value = "-FLT_MAX";
        type_max_value = "FLT_MAX";
        type_0_value = "0.0";
        type_1_value = "1.0";
    }
    else if( type == "double" )
    {
        type_min_value = "-DBL_MAX";
        type_max_value = "DBL_MAX";
        type_0_value = "0.0";
        type_1_value = "1.0";
    }
        
    else if( type == "int8_t" )
    {
        type_min_value = "INT8_MIN";
        type_max_value = "INT8_MAX";
        type_0_value = "0";
        type_1_value = "1";
    }
    else if( type == "uint8_t" )
    {
        type_min_value = "0";
        type_max_value = "UINT8_MAX";
        type_0_value = "0";
        type_1_value = "1";
    }
    else if( type == "int32_t" )
    {
        type_min_value = "INT32_MIN";
        type_max_value = "INT32_MAX";
        type_0_value = "0";
        type_1_value = "1";
    }
    else if( type == "int64_t" )
    {
        type_min_value = "INT64_MIN";
        type_max_value = "INT64_MAX";
        type_0_value = "0";
        type_1_value = "1";
    }
    else
    {
        ERROR("Unimplemented: data type " << type);
    }

    if(op_name == "ReduceProd") {
        elemet_operation = [](const std::string& a, const std::string& b)
        { return a + "*=" + b;};
        initial_value = type_1_value;
    } else if (op_name == "ReduceSum"){
        elemet_operation = [](const std::string& a, const std::string& b)
        { return a + "+=" + b;};
        initial_value = type_0_value;
    }
    else if (op_name == "ReduceL1"){
        elemet_operation = [this](const std::string& a, const std::string& b)
        { return a + "+= " + math_func("fabs") + "(" + b + ")";};
        initial_value = type_0_value;
    }
    else if (op_name == "ReduceL2"){
        elemet_operation = [](const std::string& a, const std::string& b)
        { return a + "+= (" + b + "*" + b + ")";};
        initial_value = type_0_value;
    }
    else if (op_name == "ReduceLogSum"){
        elemet_operation = [](const std::string& a, const std::string& b)
        { return a + "+=" + b;};
        initial_value = type_0_value;
    }
    else if (op_name == "ReduceLogSumExp"){
        elemet_operation = [this](const std::string& a, const std::string& b)
        { return a + "+= " + math_func("exp") + "(" + b + ")";};
        initial_value = type_0_value;
    }
    else if (op_name == "ReduceMean"){
        elemet_operation = [](const std::string& a, const std::string& b)
        { return a + "+=" + b;};
        initial_value = type_0_value;
    }
    else if (op_name == "ReduceSumSquare"){
        elemet_operation = [](const std::string& a, const std::string& b)
        { return a + "+= ("+b+" * "+b+")";};
        initial_value = type_0_value;
    } 
    else if (op_name == "ReduceMax"){
        elemet_operation = [](const std::string& a, const std::string& b)
        { return a + " = MAX(" + a + ", " + b + ")";};
        initial_value = type_min_value;
    } 
    else if (op_name == "ReduceMin"){
        elemet_operation = [](const std::string& a, const std::string& b)
        { return a + " = MIN(" + a + ", " + b + ")";};
        initial_value = type_max_value;
    } 
    else {
        ERROR("Unimplemented: node operation " << op_name);
    }

	// If axes is empty, we are reducing across all dimensions
    if (axes.empty()) {
        if (keepdims) {
            // If we keep dimensions, the shape should be all ones with the same rank as input
            t->data_dim = std::vector<int>(input->data_dim.size(), 1);
        } else {
            // If we do not keep dimensions, the result is a scalar represented by an empty shape
            t->data_dim = std::vector<int>(1, 1);
        }
    } else {
        t->data_dim = input->data_dim;
        norm_axes = normalized_axes(t);
        for (size_t axis: norm_axes) {
			// Set the specified normalized axis to 1 in the output shape
			t->data_dim[axis] = 1;
		}
        
        // If keepdims is 0, remove the dimensions set to 1 from the shape
        if (!keepdims) {
            // ERROR("Reduce with keepdims=0 not implemented yet");
            std::vector<int> new_shape;
            for (int dim_size : t->data_dim) {
                if (dim_size != 1) {
                    new_shape.push_back(dim_size);
                }
            }
            // If all dimensions were reduced, output is a scalar
            if (new_shape.empty()) {
                new_shape.push_back(1);
            }
            t->data_dim = new_shape;
        }
    }
    // Set the data type for the output tensor
    t->data_type = input->data_type;
	output = t;
	register_output(t, "y");
}


void Reduce::print(std::ostream &dst) const
{
    const Tensor *input = get_input_tensor(0);
    std::string type = input->data_type_str();

    INDT_1 << "/* "<< op_name << " */" << std::endl;
    INDT_1 << "/* keepdims: " << keepdims << " */"<< std::endl;
    INDT_1 << "/* axes: (";
    for (size_t i = 0; i < axes.size(); ++i) {
        INDT_1 << axes[i] << (i < axes.size() - 1 ? ", " : "");
    }
    INDT_1 << ") */" << std::endl;

    /* iterating only over output values for initlaization purpose*/
    std::string out_idx = print_and_return_o_iterator(dst);
    INDT(output->rank()+1) << "y" << out_idx << " = " << initial_value << ";" << std::endl;
    print_loop_closes_over_dims(dst, output->rank());
    
    /* actual operation e.g. product in case of ReduceProd*/
    std::pair<std::string, std::string> index_pair = print_and_return_io_iterator(dst);
    std::string inp_idx = index_pair.first;
    out_idx = index_pair.second;
    INDT(input->rank()+1) << elemet_operation("y" + out_idx, "x" + inp_idx) << ";" << std::endl;
    print_loop_closes_over_dims(dst, input->rank());

    /* Some derivates of Reduce require to iterate over the outputs in the end - or can it be all somehow merged?*/
    if (op_name=="ReduceMean") {
        INDT_1 << "/* ReduceMean: Divide by the number of elements (reduced axes) */" << std::endl;
        int out_size = get_number_of_reduced_elements();
        std::string out_idx = print_and_return_o_iterator(dst);
        INDT(output->rank()+1) << "y" << out_idx << " /= " << out_size << ";" << std::endl;
        print_loop_closes_over_dims(dst, output->rank());
    }
    if (op_name=="ReduceL2") {
        INDT_1 << "/* ReduceL2: Sqrt of summed sqares*/" << std::endl;
        std::string out_idx = print_and_return_o_iterator(dst);
        INDT(output->rank()+1) << "y" << out_idx << " = sqrt(y" << out_idx << ");" << std::endl;
        print_loop_closes_over_dims(dst, output->rank());
    }
    if (op_name=="ReduceLogSum" || op_name=="ReduceLogSumExp") {
        INDT_1 << "/* ReduceLogSum : log of reduced sum*/" << std::endl;
        std::string out_idx = print_and_return_o_iterator(dst);
        INDT(output->rank()+1) << "y" << out_idx << " = " << math_func("log") << "(y" << out_idx << ");" << std::endl;
        print_loop_closes_over_dims(dst, output->rank());
    }
}
}
