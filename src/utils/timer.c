#include "timer.h"
#include <time.h>

double now_seconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

double get_time() {
    LARGE_INTEGER frequency;
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / frequency.QuadPart;
}