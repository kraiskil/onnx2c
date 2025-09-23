/*
 * Generate C source from an ONNX "standard" test directory.
 * (see e.g. onnx/onnx/backend/test/data/node/test_add/ for an example)
 * This takes as paramters the directory where the test is and 
 * which of the test inputs in that test directory to use.
 * (see the "Usage" error print at start of main()).
 * The final test source is printed out on stdout.
 */


#include <iostream>
#include <fstream>

#include "graph.h"
#include "onnx.pb.h"
#include "options.h"
#include "tensor.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>

using namespace toC;

struct onnx2c_opts options;

bool load_input_data(const std::string &filename, onnx::TensorProto &result)
{

	/* TODO: read Protobuffers documentation. This is lifted from
	 * ONNX code - looks like there could be a more C++ way to do this */ 
	FILE *f = fopen(filename.c_str(), "rb");
	if( f == NULL )
		return false;
	fseek(f, 0, SEEK_END);
	ssize_t size = ftell(f);
	fseek(f, 0, SEEK_SET);

	// TODO: check if data is copied in the protobuf object.
	// If not, we leak memory here, but that's not a problem for the test suite.
	// Probably not, since this worked, and data used to be a VLA on the stack.
	char *data =new char[size];
	int nread = fread(data, 1, size, f);
	fclose(f);

	if( nread != size )
		ERROR("Problem reading input data");

	::google::protobuf::io::ArrayInputStream input_stream(data, size);
	::google::protobuf::io::CodedInputStream coded_stream(&input_stream);
	return result.ParseFromCodedStream(&coded_stream);
}

Tensor * get_input_from_file( std::string &partial_path, int input_number )
{
	onnx::TensorProto tensor;
	std::string input_fn = partial_path + std::to_string(input_number) + ".pb";
	
	if( load_input_data(input_fn, tensor) == false )
		return NULL;

	Tensor *t = new Tensor;
	t->parse_onnx_tensor(tensor);
	return t;
}

int main(int argc, char *argv[])
{
	if( argc < 4 ) {
		std::cerr << "Usage:" << std::endl;
		std::cerr << "./onnx_backend_tests_runner <directory> <accuracy> <test_data_set>" << std::endl;
		std::cerr << std::endl;
		std::cerr << " <directory> is the directory that contains the test - i.e. 'model.onnx' and test_data_set_0" << std::endl;
		std::cerr << " <accuracy> floating point value: the maximum allowed difference between result and refrence. Use decimal dot, not comma!"<< std::endl;
		std::cerr << " <test_data_set> integer value: select the test dataset to run this test against. (Most tests have only 0)" << std::endl;
		exit(1);
	}

	options.logging_level = 1;
	AixLog::Log::init<AixLog::SinkCerr>(AixLog::Severity::error);

	onnx::ModelProto onnx_model;
	std::string dir(argv[1]);

	// Beware of user having locale support on! Test suite feeds floats with the decimal dot format.
	setlocale(LC_NUMERIC, "C");
	float test_accuracy = std::stod(argv[2]);

	std::vector<Tensor*> inputs;
	std::vector<Tensor*> outputs;
	std::vector<Tensor*> references;

	std::string dataset_dir = dir + "/test_data_set_" + argv[3];


	int input_number=0;
	while(true) {
		std::string partial = dataset_dir + "/input_";
		Tensor *t = get_input_from_file(partial, input_number);
		if( t == NULL )
			break;
		t->isIO = true;
		// Don't write the initialization from the onnx2c graph
		// It is written from the test suite, which is part of "the application",
		// not the neural net
		t->initialize = false;
		t->generate = false;
		t->isConst=true;
		if( t->name == "" )
			t->name = std::string("input_") + std::to_string(input_number);
	
		inputs.push_back(t);
		input_number++;
	}

	input_number=0;
	while(true) {
		std::string partial = dataset_dir + "/output_";
		Tensor *ref = get_input_from_file(partial, input_number);
		Tensor *out = get_input_from_file(partial, input_number);
		if( ref == NULL || out == NULL )
			break;
		ref->generate=true;
		ref->initialize=true;
		out->generate=true;
		out->initialize=false;
		out->isConst=false;
		// Just in case the network needs to zero-initialize the output buffer (e.g. LSTM)
		out->data_buffer = NULL;
		out->isIO = true;
		if( ref->name == "" ) {
			ref->name = std::string("output_") + std::to_string(input_number);
			out->name = std::string("output_") + std::to_string(input_number);
		}

		references.push_back(ref);
		outputs.push_back(out);
		input_number++;
	}


	// Read in model
	std::string model_fn = dir + "/model.onnx";
	std::ifstream model_ifs(model_fn);
	if (!model_ifs.good()) {
		std::cerr << "Error opening model file: " << model_fn << std::endl;
		exit(1); //TODO: check out error numbers for a more accurate one
	}

	// We pass in the testsuite's tensors to the Graph, so it
	// can mark IO tensors as 'initialized'.
	// This helps with unittests where the node expects input to be
	// a compile time constant (e.g. Unsqueeze)
	std::vector <Tensor *> tensors_to_parser;
	for( auto i : inputs) tensors_to_parser.push_back(i);

	onnx_model.ParseFromIstream(&model_ifs);
	Graph toCgraph(onnx_model, tensors_to_parser);

	// Optionally, genrerate the network into the same file as the test harness.
	// Alternatively, it gets generated by onnx2c, and linked in.
	// This latter way is more correct, as it reflects how the end user would use onnx2c.
	// The former way neeeded because the unittests are a bit degenerate. E.g.
	// shape, reshape and const nodes have corner cases where the one or
	// the other approach is more appropriate. (e.g. the unit tests return
	// a constant tensor, or the nodes expect the input to be compile time
	// constants)
#if defined TESTGEN_SINGLEFILE
	std::cout.precision(20);
	toCgraph.unionize_tensors();
	toCgraph.print_source(std::cout);
	std::cout << std::endl << std::endl;
	std::cout << "/////////////////////////////////////"<<std::endl;
	std::cout << "// End of compiled graph."<<std::endl;
	std::cout << "// All stuff below is test suite code"<<std::endl;
	std::cout << "/////////////////////////////////////"<<std::endl;
	std::cout << std::endl << std::endl;
#else
	std::cout << "#include <float.h>"<<std::endl;
	std::cout << "#include <math.h>"<<std::endl;
	std::cout << "#include <stdbool.h>"<<std::endl;
	std::cout << "#include <stdint.h>"<<std::endl;
	toCgraph.print_interface_function(std::cout, false); // false==declaration
#endif

	for( auto i : inputs) {
		std::string refname = "graphin_" + i->cname();
		std::cout << "static ";
		std::cout << i->print_tensor_definition(refname);
		std::cout << " = ";
		i->print_tensor_initializer(std::cout);
		std::cout << ";" << std::endl;
	}
	for( auto o : outputs) {
		std::string refname = "graphout_" + o->cname();
		std::cout << "static ";
		std::cout << o->print_tensor_definition(refname);
		std::cout << ";" << std::endl;
	}
	// print the reference tensors
	for( auto o : references ) {
		std::string refname = "reference_" + o->cname();
		std::cout << "static ";
		std::cout << o->print_tensor_definition(refname);
		std::cout << " = ";
		o->print_tensor_initializer(std::cout);
		std::cout << ";" << std::endl;
	}


	std::cout <<         "int main(void) {" << std::endl;

	// print the call to the "entry()" function that
	// run inference on the network
	std::cout << "\t"<<  "entry(";
	bool isfirst = true;
	for( auto i : inputs) {
		if( isfirst ) isfirst=false;
		else          std::cout << ", ";
		if( i->is_scalar() )
			std::cout << "&";
		std::cout << "graphin_" + i->cname();
	}
	for( auto r : outputs ) {
		if( isfirst ) isfirst=false;
		else          std::cout << ", ";
		if( r->is_scalar() )
			std::cout << "&";
		std::cout << "graphout_"+r->cname();
	}
	std::cout << ");" << std::endl;
	std::cout << std::endl;


	// Loop over outuputs
	for( unsigned i=0; i<outputs.size(); i++ ) {
	std::cout << "\t{" << std::endl;
		Tensor *r = references[i];
		Tensor *o = outputs[i];
		std::string type = r->data_type_str();

		// variable for the result
		std::string outname = "graphout_" + o->cname();
		std::cout << "\t\t" << type << " *result = (" << type << "*)";
		if( o->is_scalar() )
			std::cout << "&";
		std::cout << outname << ";" << std::endl;

		// variable for the reference
		std::string refname = "reference_" + r->cname();
		std::cout << "\t\t" << type << " *reference = (" << type << "*)";
		if( r->is_scalar() )
			std::cout << "&";
		std::cout << refname << ";" << std::endl;

		// Check result and reference, elementvise
		std::cout << "\t\t" << "for(uint64_t i = 0; i< (sizeof(" << refname << ") / sizeof("<<type<<")); i++) {" << std::endl;
		if( type == "float" || type == "double" ) {
			std::cout << "\t\t\t" << "if( fabs(result[i]-reference[i]) > " << test_accuracy << " )" <<std::endl;
			std::cout << "\t\t\t\t" << "return 1;" << std::endl;
			// fabs(nan) > 0.1 always false - and out-of-bounds indexing is a likely bug and source of nans
			std::cout << "\t\t\t" << "if(isnan(result[i]) || isnan(reference[i]))" << std::endl;
			std::cout << "\t\t\t\t" << "return 1;" << std::endl;
		}
		else if(   type == "int8_t"
		        || type == "uint8_t"
		        || type == "int16_t"
		        || type == "uint16_t"
		        || type == "int32_t"
		        || type == "uint32_t"
		        || type == "int64_t"
		        || type == "uint64_t"
			|| type == "bool" ) {
			std::cout << "\t\t\t" << "if( result[i] != reference[i] )" <<std::endl;
			std::cout << "\t\t\t\t" << "return 1;" << std::endl;
			// no nan checking needed
		}
		else
			ERROR("unimplemented type");
		std::cout << "\t}" << std::endl;
	std::cout << "\t}" << std::endl;
	}

	std::cout << "\treturn 0;" << std::endl;
	std::cout << "}" << std::endl;
	return 0;
}

