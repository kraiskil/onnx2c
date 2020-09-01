#include "aixlog.hpp"
#include <cassert>

#define ERROR(why) do { \
	LOG(FATAL) << why; \
	assert(false); \
	exit(1);\
} while(0)

