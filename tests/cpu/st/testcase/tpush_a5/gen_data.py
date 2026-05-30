#!/usr/bin/env python3
import numpy as np
import os

# Generate random inputs
a = np.random.randn(32, 64).astype(np.float32)
b = np.random.randn(64, 512).astype(np.float32)
c_prev = np.random.randn(32, 512).astype(np.float32)

# Compute golden output: c = c_prev + matmul(a + 1, b)
c_golden = c_prev + np.matmul(a + 1.0, b)

# Save as raw binary files

for i in range(4):
    case_name = f"TPUSH_A5Test.case_{i+1}"
    if i == 3:
        c_golden = c_prev + np.matmul(a, b)
    if not os.path.exists(case_name):
        os.makedirs(case_name)
    original_dir = os.getcwd()
    os.chdir(case_name)

    a.tofile("a.bin")
    b.tofile("b.bin")
    c_prev.tofile("c.bin")
    c_golden.tofile("golden.bin")

    os.chdir(original_dir)
