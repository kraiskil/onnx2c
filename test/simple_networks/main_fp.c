#include <stdio.h>
#include <math.h>

extern unsigned count;
extern float reference[];
extern float result[];

void entry(float*);

int main() {
    entry(result);

    for (unsigned i = 0; i < count; i++) {
        if( isnan(result[i]) != isnan(reference[i]) ||
            isinf(result[i]) != isinf(reference[i]) ||
            (isfinite(result[i]) && fabs(result[i] - reference[i]) > 1e-5) ||
            (isinf(result[i]) && (result[i] < 0) != (reference[i] < 0)) ) {
            printf("Mismatch at index %u: result=%f, reference=%f\n", i, result[i], reference[i]);
            return 1;
        }
    }
    return 0;
}