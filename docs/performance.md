### Baseline Observation

The naive matrix multiplication showed poor performance for
512×512 matrices, completing in ~0.56 seconds. This implementation
exhibits poor cache locality due to repeated access of matrix B
across rows of A.
