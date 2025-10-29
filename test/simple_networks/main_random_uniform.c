#include <stdio.h>
#include <math.h>

#define SIZE (3 * 4 * 5)

float output[SIZE];

void entry(float*);

int main() {
    entry(output);
    for (unsigned i = 0; i < SIZE; i++) {
        if( isnan(output[i]) )
            return 1;
        if( output[i] < 0.0f || output[i] > 1.0f )
            return 1;
    }
    return 0;
}
