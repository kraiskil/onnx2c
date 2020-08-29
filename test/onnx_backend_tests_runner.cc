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
	fread(data, size, 1, f);
	fclose(f);

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
		Tensor *t = get_input_from_file(partial, input_number);
		if( t == NULL )
			break;
		t->generate=true;
		t->initialize=false;
		t->isIO = true;
		if( t->name == "" )
			t->name = std::string("output_") + std::to_string(input_number);

		// TODO: rest of the framework does not handle multiple outputs yet
		if( input_number == 1 )
			break;
		references.push_back(t);
		input_number++;
	}

	if( references.size() != 1 ) {
		std::cerr << "Only one output implemented for now" << std::endl;
		exit(1);
	}

	std::string model_fn = dir + "/model.onnx";
	std::ifstream model_ifs(model_fn);
	if (!model_ifs.good()) {
		std::cerr << "Error opening modele file: " << model_fn << std::endl;
		exit(1); //TODO: check out error numbers for a more accurate one
	}

	std::vector <Tensor *> tensors_to_parser;
	for( auto i : inputs) tensors_to_parser.push_back(i);
	for( auto i : references) tensors_to_parser.push_back(i);

	onnx_model.ParseFromIstream(&model_ifs);
	Graph toCgraph(onnx_model, tensors_to_parser);
	toCgraph.print_source(std::cout);


	for( auto o : references ) {
		std::string refname = "reference_" + o->cname();
		std::cout << "static ";
		o->print_type_name_dimensions(std::cout, "reference_");
		std::cout << " = ";
		o->print_tensor_initializer(std::cout);
		std::cout << ";" << std::endl;
	}

	std::string refname = "reference_" + references[0]->cname();
	std::string type = references[0]->data_type_str();


	std::cout <<         "int main(void) {" << std::endl;
	std::cout << "\t" << type << " *result = (" << type << "*)" << references[0]->cname() << ";" << std::endl;
	std::cout << "\t" << type << " *reference = (" << type << "*)" << refname << ";" << std::endl;
	std::cout << "\t"<<  "entry(";
	for( auto i = inputs.begin(); i<inputs.end(); i++) {
		std::cout << (*i)->cname();
		std::cout << ", ";
	}
	std::cout <<  references[0]->cname() << ");" << std::endl;
	std::cout << std::endl;
	std::cout << "\t" << "for(uint64_t i = 0; i< (sizeof(" << refname << ") / sizeof("<<type<<")); i++) {" << std::endl;
	if( type == "float" || type == "double" ) {
		std::cout << "\t\t" << "if( fabs(result[i]-reference[i]) > " << test_accuracy << " )" <<std::endl;
		std::cout << "\t\t\t" << "return 1;" << std::endl;
		// fabs(nan) > 0.1 always false - and out-of-bounds indexing is a likely bug and source of nans
		std::cout << "\t\t" << "if(isnan(result[i]) || isnan(reference[i]))" << std::endl;
		std::cout << "\t\t\t" << "return 1;" << std::endl;
	}
	else if( type == "uint8_t" ) {
		std::cout << "\t\t" << "if( result[i] != reference[i] )" <<std::endl;
		std::cout << "\t\t\t" << "return 1;" << std::endl;
		// no nan checking needed
	}
	std::cout << "\t}" << std::endl;

	std::cout << "\treturn 0;" << std::endl;
	std::cout << "}" << std::endl;
	return 0;
}

