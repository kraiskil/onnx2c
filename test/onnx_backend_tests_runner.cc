#include <iostream>
#include <fstream>

#include "graph.h"
#include "onnx.pb.h"
#include "tensor.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>

using namespace toC;


bool load_input_data(const std::string &filename, onnx::TensorProto &result)
{

	/* TODO: read Protobuffers documentation. This is lifted from
	 * ONNX code - looks like there could be a more C++ way to do this */ 
	FILE *f = fopen(filename.c_str(), "rb");
	if( f == NULL )
		return false;
	fseek(f, 0, SEEK_END);
	int size = ftell(f);
	fseek(f, 0, SEEK_SET);

	char data[size];
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

	// Why do we send input and reference tensors to the Graph?
	std::vector <Tensor *> tensors_to_parser;
	for( auto i : inputs) tensors_to_parser.push_back(i);
	for( auto i : outputs) tensors_to_parser.push_back(i);

	onnx_model.ParseFromIstream(&model_ifs);
	Graph toCgraph(onnx_model, false/*verbose*/, tensors_to_parser);
	std::cout.precision(20);
	toCgraph.print_source(std::cout);

	// print the reference tensors
	for( auto o : references ) {
		std::string refname = "reference_" + o->cname();
		std::cout << "static ";
		if( o->isAliasOf )
			o->isAliasOf->print_tensor(std::cout, false, refname);
		else
			o->print_tensor(std::cout, false, refname);
		std::cout << " = ";
		o->print_tensor_initializer(std::cout);
		std::cout << ";" << std::endl;
	}


	std::cout <<         "int main(void) {" << std::endl;

	// run inference on the network
	std::cout << "\t"<<  "entry(";
	bool isfirst = true;
	for( auto i : inputs) {
		if( i-> isAliasOf )
			continue;
		if( isfirst ) isfirst=false;
		else          std::cout << ", ";
		std::cout << i->cname();
	}
	for( auto r : outputs ) {
		if( r->isAliasOf )
			continue;
		std::cout << ", ";
		std::cout << r->cname();
	}
	std::cout << ");" << std::endl;
	std::cout << std::endl;


	// Loop over outuputs
	for( unsigned i=0; i<outputs.size(); i++ ) {
	std::cout << "\t{" << std::endl;
		Tensor *r = references[i];
		Tensor *o = outputs[i];
		std::string outname = o->isAliasOf? o->isAliasOf->cname() : o->cname();
		std::string refname = "reference_" + r->cname();
		std::string type = r->data_type_str();

		std::cout << "\t\t" << type << " *result = (" << type << "*)" << outname << ";" << std::endl;
		std::cout << "\t\t" << type << " *reference = (" << type << "*)" << refname << ";" << std::endl;

		// Check result and reference, elementvise
		std::cout << "\t\t" << "for(uint64_t i = 0; i< (sizeof(" << refname << ") / sizeof("<<type<<")); i++) {" << std::endl;
		if( type == "float" || type == "double" ) {
			std::cout << "\t\t\t" << "if( fabs(result[i]-reference[i]) > " << test_accuracy << " )" <<std::endl;
			std::cout << "\t\t\t\t" << "return 1;" << std::endl;
			// fabs(nan) > 0.1 always false - and out-of-bounds indexing is a likely bug and source of nans
			std::cout << "\t\t\t" << "if(isnan(result[i]) || isnan(reference[i]))" << std::endl;
			std::cout << "\t\t\t\t" << "return 1;" << std::endl;
		}
		else if( type == "uint8_t" || type == "int64_t" ) {
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

