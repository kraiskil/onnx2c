#include <iostream>

#define ERROR(why) do { \
	std::cerr << why << ". At " << __FILE__ << ":" << __LINE__ << std::endl;\
	exit(1);\
} while(0)
