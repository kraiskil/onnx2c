#include "aixlog.hpp"

#define ERROR(why) do { \
	LOG(FATAL) << why; \
	exit(1);\
} while(0)

