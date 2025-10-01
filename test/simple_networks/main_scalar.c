#include <stdio.h>
#include <math.h>

extern unsigned count;
extern float input[];
extern float reference[];

float result = 0.0f;

void entry(const float*, float*);

int main() {
    for (unsigned i = 0; i < count; i++) {
        entry(&input[i], &result);

        if( isnan(result) )
            return 1;
        if( fabs(result - reference[i]) > 1e-5 )
            return 1;
    }
    return 0;
}