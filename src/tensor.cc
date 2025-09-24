#include "tensor.h"
#include "util.h"
#include <limits>

using namespace toC;
void Tensor::parse_onnx_tensor(const onnx::TensorProto &tensor)
{

	generate=true;
	initialize=true;
	isIO = false;
	isConst = true;

	// assert tensor is resolvable
	if( onnx::TensorProto_DataLocation() != onnx::TensorProto_DataLocation_DEFAULT )
		ERROR("unhandled: non-default data location in tensor " << tensor.name());
	if( tensor.has_segment() )
		ERROR("unhandled: segmented data in tensor" << tensor.name());


	int32_t datatype = tensor.data_type();
	if( onnx::TensorProto_DataType_IsValid(datatype) == false )
		ERROR("Non-valid data type " << datatype << " in tensor " << tensor.name());
	data_type = static_cast<onnx::TensorProto_DataType>(datatype);

	// Number of data in the ONNX protobuffer. Except if data is stored "externally" this will be 0 :|
	int data_num_elements;
	switch( datatype )
	{
		case onnx::TensorProto_DataType_FLOAT:
			data_num_elements = tensor.float_data_size(); break;
		case onnx::TensorProto_DataType_DOUBLE:
			data_num_elements = tensor.double_data_size(); break;

		// NB: all datatypes of 32bit or less are contained in int32_data field
		case onnx::TensorProto_DataType_BOOL:
			data_num_elements = tensor.int32_data_size(); break;
		case onnx::TensorProto_DataType_INT8:
			data_num_elements = tensor.int32_data_size(); break;
		case onnx::TensorProto_DataType_UINT8:
			data_num_elements = tensor.int32_data_size(); break;
		case onnx::TensorProto_DataType_INT16:
			data_num_elements = tensor.int32_data_size(); break;
		case onnx::TensorProto_DataType_UINT16:
			data_num_elements = tensor.int32_data_size(); break;
		case onnx::TensorProto_DataType_INT32:
			data_num_elements = tensor.int32_data_size(); break;
		case onnx::TensorProto_DataType_UINT32:
			data_num_elements = tensor.int32_data_size(); break;
		case onnx::TensorProto_DataType_INT64:
			data_num_elements = tensor.int64_data_size(); break;
		case onnx::TensorProto_DataType_UINT64:
			data_num_elements = tensor.uint64_data_size(); break;
		default:
			ERROR("unhandled tensor data type in tensor " << tensor.name());
			break;
	};

	int64_t calc_num_data =1;
	for( int dim : tensor.dims() ) {
		data_dim.push_back(dim);
		calc_num_data *= dim;
	}
	if( data_num_elements != calc_num_data ) {
		if( data_num_elements != 0 )
			ERROR("Error: data size does not match dimensions, and data_num_elem is not zero");
		else if( tensor.has_raw_data() == false )
			ERROR("Error: data size does not match dimensions, and no raw data");
	}

	data_buffer = malloc(data_num_elem() * data_elem_size());
	if( data_buffer == NULL )
		ERROR("memory allocation failed for tensor " << tensor.name());

	if( tensor.has_raw_data() ) {
		std::string raw_data = tensor.raw_data(); // Yes, std::string!
		if( raw_data.size() != (uint64_t)(calc_num_data*data_elem_size()) )
			ERROR("Error: tensor raw data size does not match dimensions");

		memcpy( data_buffer, raw_data.c_str(), raw_data.size() );
	}

	else {
		switch( datatype )
		{
			// NB: all datatypes of 32bit or less are contained in int32_data field
			// The documentation is not quite clear on this, but this passes the tests.
			case onnx::TensorProto_DataType_INT8:
				for( int i=0; i<data_num_elem(); i++  )
					((int8_t*)data_buffer)[i] = tensor.int32_data(i);
				break;
			case onnx::TensorProto_DataType_UINT8:
				for( int i=0; i<data_num_elem(); i++  )
					((uint8_t*)data_buffer)[i] = tensor.int32_data(i);
				break;
			case onnx::TensorProto_DataType_INT16:
				for( int i=0; i<data_num_elem(); i++  )
					((int16_t*)data_buffer)[i] = tensor.int32_data(i);
				break;
			case onnx::TensorProto_DataType_UINT16:
				for( int i=0; i<data_num_elem(); i++  )
					((uint16_t*)data_buffer)[i] = tensor.int32_data(i);
				break;
			case onnx::TensorProto_DataType_INT32:
				for( int i=0; i<data_num_elem(); i++  )
					((int32_t*)data_buffer)[i] = tensor.int32_data(i);
				break;
			case onnx::TensorProto_DataType_UINT32:
				for( int i=0; i<data_num_elem(); i++  )
					((uint32_t*)data_buffer)[i] = tensor.int32_data(i);
				break;
			case onnx::TensorProto_DataType_INT64:
				for( int i=0; i<data_num_elem(); i++  )
					((int64_t*)data_buffer)[i] = tensor.int64_data(i);
				break;
			case onnx::TensorProto_DataType_UINT64:
				for( int i=0; i<data_num_elem(); i++  )
					((uint64_t*)data_buffer)[i] = tensor.uint64_data(i);
				break;
			case onnx::TensorProto_DataType_FLOAT:
				for( int i=0; i<data_num_elem(); i++  )
					((float*)data_buffer)[i] = tensor.float_data(i);
				break;
			default:
				ERROR("unhandled tensor data type in tensor " << tensor.name());
				break;
		};
	}

	name = tensor.name();
	doc = tensor.doc_string();

}

std::string Tensor::cname(void) const
{
	return "tensor_" + cify_name(name);
}

int Tensor::data_elem_size(void)const
{
	switch( data_type )
	{
	case onnx::TensorProto_DataType_FLOAT:
			return sizeof(float); break;
		case onnx::TensorProto_DataType_DOUBLE:
			return sizeof(double); break;
		case onnx::TensorProto_DataType_INT8:
			return sizeof(int8_t); break;
		case onnx::TensorProto_DataType_UINT8:
			return sizeof(uint8_t); break;
		case onnx::TensorProto_DataType_INT16:
			return sizeof(int16_t); break;
		case onnx::TensorProto_DataType_UINT16:
			return sizeof(uint16_t); break;
		case onnx::TensorProto_DataType_INT32:
			return sizeof(int32_t); break;
		case onnx::TensorProto_DataType_UINT32:
			return sizeof(uint32_t); break;
		case onnx::TensorProto_DataType_INT64:
			return sizeof(int64_t); break;
		case onnx::TensorProto_DataType_UINT64:
			return sizeof(uint64_t); break;
		case onnx::TensorProto_DataType_BOOL:
			return sizeof(bool); break;
		default:
			ERROR("unhandled tensor data type in tensor " << name);
			break;
	};
}

std::string Tensor::data_type_str(void) const
{
	switch( data_type )
	{
		case onnx::TensorProto_DataType_FLOAT:
			return "float"; break;
		case onnx::TensorProto_DataType_DOUBLE:
			return "double"; break;
		case onnx::TensorProto_DataType_INT8:
			return "int8_t"; break;
		case onnx::TensorProto_DataType_UINT8:
			return "uint8_t"; break;
		case onnx::TensorProto_DataType_INT16:
			return "int16_t"; break;
		case onnx::TensorProto_DataType_UINT16:
			return "uint16_t"; break;
		case onnx::TensorProto_DataType_INT32:
			return "int32_t"; break;
		case onnx::TensorProto_DataType_UINT32:
			return "uint32_t"; break;
		case onnx::TensorProto_DataType_INT64:
			return "int64_t"; break;
		case onnx::TensorProto_DataType_UINT64:
			return "uint64_t"; break;
		case onnx::TensorProto_DataType_BOOL:
			return "bool"; break;
		case onnx::TensorProto_DataType_UNDEFINED:
			return "UNDEFINED"; break;
		default:
			ERROR("unhandled tensor data type in tensor " << name);
			break;
	};
}


void Tensor::print_element(std::ostream &dst, uint64_t element) const
{
	switch(data_type)
	{
		case onnx::TensorProto_DataType_FLOAT:
		{
			/*
			some tests require large number e.g. 479001600
			using std::showpoint prints 4.79002e+08f
			The test passes if std::fixed is used printing 479001600.000000
			*/
			float *f = static_cast<float*>(data_buffer);
			dst << std::fixed << f[element]<< "f";
			break;
		}
		case onnx::TensorProto_DataType_DOUBLE:
		{
			double *f = static_cast<double*>(data_buffer);
			dst << std::fixed << f[element]<< "f";
			break;
		}
		case onnx::TensorProto_DataType_INT8:
		{
			int8_t *f = static_cast<int8_t*>(data_buffer);
			// don't print as characters
			dst << static_cast<int>(f[element]);
			break;
		}
		case onnx::TensorProto_DataType_UINT8:
		{
			uint8_t *f = static_cast<uint8_t*>(data_buffer);
			// don't print as characters
			dst << static_cast<int>(f[element]);
			break;
		}
		case onnx::TensorProto_DataType_INT16:
		{
			int16_t *f = static_cast<int16_t*>(data_buffer);
			dst << f[element];
			break;
		}
		case onnx::TensorProto_DataType_UINT16:
		{
			uint16_t *f = static_cast<uint16_t*>(data_buffer);
			dst << f[element];
			break;
		}
		case onnx::TensorProto_DataType_INT32:
		{
			int32_t *f = static_cast<int32_t*>(data_buffer);
			dst << f[element];
			break;
		}
		case onnx::TensorProto_DataType_UINT32:
		{
			uint32_t *f = static_cast<uint32_t*>(data_buffer);
			dst << f[element];
			break;
		}
		case onnx::TensorProto_DataType_INT64:
		{
			int64_t *f = static_cast<int64_t*>(data_buffer);
			dst << f[element];
			break;
		}
		case onnx::TensorProto_DataType_UINT64:
		{
			uint64_t *f = static_cast<uint64_t*>(data_buffer);
			dst << f[element];
			break;
		}
		case onnx::TensorProto_DataType_BOOL:
		{
			bool *f = static_cast<bool*>(data_buffer);
			dst << f[element];
			break;
		}

		default:
			ERROR("unimplemented printing of initialized datatype " << data_type_str());
	}
}

/* Print the tensor initialization values.
 * This is the values between '=' and ';'.
 * The function recurses into itself to print multidimensional initializers.
 *
 * dim: the dimension from which to print.
 * offs: the offset into this dimension from where to print.
 * This function recurses back into itself to print all more inner dimenstions there are.
 * I.e. if calling with dim=0, offs=0 (which are default values),
 * it prints the entire variable initialzation.
 */
void Tensor::print_tensor_initializer(std::ostream &dst, int dim, int offs) const
{
	if( is_scalar() )
	{
		print_element(dst, offs);
		return;
	}

	if( data_dim[dim] == 0 )
		return;

	for( int i=0; i<dim; i++)
		dst << "  ";

	dst << "{" ;

	// if this is printing "outer" dimensions, recurse back in till we hit
	// the innermost dimension
	if(   dim < (int)(data_dim.size()-1) ) {
		dst << std::endl;
		for( int i=0; i<data_dim[dim]; i++ )
		{
			int remaining_dims=1;
			for(unsigned j = dim+1; j<data_dim.size(); j++)
				remaining_dims *= data_dim[j];
			print_tensor_initializer(dst, dim+1, offs+i*remaining_dims);
			if( i <(data_dim[dim]-1) )
				dst << ",";
			dst << std::endl;
		}
		// indent a stand-alone closing brace
		for( int i=0; i<dim; i++)
			dst << "  ";
	}

	else {
		for( int i=0; i<data_dim[dim]; i++)
		{
			int element=offs+i;
			print_element(dst, element);
			if( i <(data_dim[dim]-1) )
				dst << ", ";
		}
	}

	dst << "}";
}


/* Print the 'float foo[N][N]' part of the tensor.
 * This is used for declaring of the tensor and as parameters to function definitions and calls.
 *
 * Outcomes for the result are:
 * - foo (callsite: i.e. for a function argument when function is called)
 * - const float foo[N] (definition - i.e. not a callsite)
 * And for scalars, it's a bit more complex:
 * - foo (definition)
 * - &foo (callsite, and the tensor is an scalar)
 * - *foo (as a parameter in a function definition)
 * */
std::string Tensor::print_tensor(
		std::string alternate_name,
		bool is_callsite,
		bool as_const,
		bool is_definition) const
{
	std::string rv = "";
	if( is_callsite == false ) {
		bool print_const = as_const || isConst;
		if( print_const )
			rv += "const ";
		rv += data_type_str() + " ";
	}
	else if( union_no >= 0 ) {
		rv += "tu" + std::to_string(union_no) + ".";
	}

	if( is_scalar() ) {
		if( is_callsite ) {
			if ( !isIO ) {
				rv += "&";
			}
		} else if ( !is_definition ) {
			rv += "*";
		}
	}

	if( alternate_name == "" )
		rv += cname();
	else
		rv += alternate_name;

	if( is_callsite == false )
		for( unsigned i : data_dim )
			rv += "[" + std::to_string(i) + "]";

	return rv;
}

int Tensor::data_num_elem(void) const
{
	int dim=1;
	for( auto i : data_dim )
		dim *= i;

	return dim;
}

unsigned Tensor::rank(void) const
{
	return data_dim.size();
}

std::string Tensor::str_dimensions(void) const
{
	std::string rv = "";
	for( auto d : data_dim ) {
		rv+= std::to_string(d);
		rv+= " ";
	}
	return rv;
}

Tensor* Tensor::make_quantized_copy(void)
{
	LOG(DEBUG) << "checking if " << name << " can be quantized" << std::endl;
	// uh... when is a tensor not quantizable? When it already is integers? When it already is 8-bit integers?
	if( data_type != onnx::TensorProto_DataType_FLOAT )
	{
		LOG(DEBUG) << "type is not float -> no quantization" << std::endl;
		return NULL;
	}
	LOG(DEBUG) << "quantizing it" << std::endl;

	Tensor *t = new Tensor();
	t->generate = generate;
	t->initialize = initialize;
	t->isConst = isConst;
	t->isIO = isIO;
	t->isRecursive = isRecursive;
	// TODO: alias?
	t->isQuantized = true;
	quantizedCopy = t;

	t->data_dim = data_dim;
	t->data_buffer = calloc(data_num_elem(), 1);
	t->name = name + "_quantized";


	/* Calculate data max and min values.
	 * TODO: do we want to quantize over channels, filter, etc. separate?
	 *       Would need more context info at this point. Maybe easier to let
	 *       the frontend NN framweworks mature a bit, and not even try to quantize in
	 *       onnx2c? But for now, do a global quantization.
	 */

	/* TODO: rewrite-this */
	if( data_type == onnx::TensorProto_DataType_FLOAT )
	{

		float maxval=-std::numeric_limits<float>::infinity();
		float minval=std::numeric_limits<float>::infinity();
		float *odata = (float*)data_buffer;
		int8_t *qdata = (int8_t*)t->data_buffer;
		t->data_type = onnx::TensorProto_DataType_INT8;

		for( int i=0; i<data_num_elem(); i++)
		{
			if( odata[i] < minval )
				minval = odata[i];
			if( odata[i] > maxval )
				maxval = odata[i];
		}
		LOG(DEBUG) << "Input tensor minval: " << minval << ", maxval: " << maxval << std::endl;

		if( maxval < 0 )
			maxval = 0;
		if( minval > 0 )
			minval = 0;

		// NB: here we cut a bit corners.
		// Assume input is centered around zero.
		// This is
		// a) probable
		// b) much easier to generate. If x and w zero_values tensors are not given, they are implicitly 0
		if( -minval > maxval )
			maxval = -minval;


		for( int i=0; i<data_num_elem(); i++) {
			float fv = (odata[i] / maxval) * 127;
			assert( fv <= 127 );
			assert( fv >= -127 );

			qdata[i] = (int8_t)fv;
			assert( odata[i] * qdata[i] >= 0 );
		}
	}
	else if( data_type == onnx::TensorProto_DataType_INT64 )
	{
		int64_t *odata = (int64_t*)data_buffer;
		uint16_t *qdata = (uint16_t*)t->data_buffer;
		t->data_type = onnx::TensorProto_DataType_UINT16;


		for( int i=0; i<data_num_elem(); i++) {
			if( odata[i] < 0 || odata[i] > 0xffff )
				ERROR("Unimplemented: quantization in this case");

			qdata[i] =(uint16_t) odata[i];
		}

	}
	else
		ERROR("Unimplemented quantization data type");

	return t;
}

bool Tensor::is_used(void) const
{
	return name != "";
}


int64_t Tensor::get_data_element(uint64_t i) const
{
	switch( data_type )
	{
		case onnx::TensorProto_DataType_INT32:
			return ((int32_t*)data_buffer)[i];
		case onnx::TensorProto_DataType_INT64:
			return ((int64_t*)data_buffer)[i];
		default:
			ERROR("Unhandled data type");
	}

	return INT64_MIN;
}
float Tensor::get_data_element_float(uint64_t i) const
{
	switch( data_type )
	{
		case onnx::TensorProto_DataType_FLOAT:
			return ((float*)data_buffer)[i];
		default:
			ERROR("Unhandled data type");
	}

	return 0;
}

std::string Tensor::print_trace_dump(void) const
{
	// TODO: there was some new pretty printing stuff in C++20?
	std::stringstream rv;
	rv << "  " <<  name << ":" // "global", ONNX name
		   << "  gen " << generate
		   << "  init " << initialize
		   << "  IO " << isIO
		   << "  const " << isConst
		   << "  recurs " << isRecursive
		   << "  dims { " << str_dimensions() << "}"
		   << "  buffer " << data_buffer
		;

	return rv.str();
}

