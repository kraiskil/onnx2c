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

void entry( float *, float *);

int main(int argc, char *argv[])
{
	if( argc < 2 ) {
		std::cerr << "provide directory with input and reference data" << std::endl;
		exit(1);
	}

	onnx::ModelProto onnx_model;
	std::string dir(argv[1]);

	std::vector<Tensor*> inputs;
	std::vector<Tensor*> references;

	// TODO: at this time, node tests have just one data set. When the time comes,
	//       loop over other test_data_set_* directories
	std::string dataset_dir = dir + "/test_data_set_0";


	int input_number=0;
	while(true) {
		std::string partial = dataset_dir + "/input_";
		Tensor *t = get_input_from_file(partial, input_number);
		if( t == NULL )
			break;
		inputs.push_back(t);
		input_number++;
	}

	input_number=0;
	while(true) {
		std::string partial = dataset_dir + "/output_";
		Tensor *t = get_input_from_file(partial, input_number);
		if( t == NULL )
			break;
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

	onnx_model.ParseFromIstream(&model_ifs);
	Graph toCgraph(onnx_model, inputs);
	toCgraph.print_source(std::cout);


#if 0
	for( auto i : inputs ) {
		std::cout << "static ";
		i->print_type_name_dimensions(std::cout);
		std::cout << " = ";
		i->print_tensor_initializer(std::cout);
		std::cout << ";" << std::endl;
	}
#endif

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
	std::cout << "\t\t" << "if( fabs(result[i]-reference[i]) > 1e-5 )" <<std::endl;
	std::cout << "\t\t\t" << "return 1;" << std::endl;
	std::cout << "\t}" << std::endl;

	std::cout << "\treturn 0;" << std::endl;
	std::cout << "}" << std::endl;
	return 0;
}

