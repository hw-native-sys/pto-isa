# Dhrystone Benchmark Report

## Introduction

Dhrystone is a synthetic benchmark program designed to measure a processor's integer performance. It was developed by Reinhold P. Weicker in 1984 and is commonly used to evaluate CPU performance in terms of integer operations, logical operations, and system calls.

This test is executed on Huawei Ascend AI processors to assess their integer computation capabilities and overall performance.

## Test Methodology

### Test Environment
- **Configuration**: Single-core test with kernel warmup

### Implementation
This test is implemented using CCE programming language with the following features:
- Functions marked with AICORE execute on AI Cores
- Template parameters enable testing with different iteration counts
- Precise timing measurement using `get_sys_cnt()` function

### Test Commands

**A2/A3 Platform:**
```bash
python3 tests/script/run_st.py -r npu -v a3 -t t_dhrystone -g TDHRYSTONETest.case_1000i -d
```

**A5 Platform:**
```bash
python3 tests/script/run_st.py -r npu -v a5 -t t_dhrystone -g TDHRYSTONETest.case_1000i -d
```

---

## A2 Platform Test Results (Ascend910B @ 1800MHz)

### Test Environment
- **Processor**: Ascend910B
- **Frequency**: 1800 MHz

### Test Data

| Iterations | Execution Time (μs) | Dhrystones/sec | DMIPS | DMIPS/MHz |
|-----------|-------------------|----------------|-------|-----------|
| 1000      | 330               | 3,030,303      | 1,724.31 | 0.958     |
| 2000      | 657               | 3,044,138      | 1,732.42 | 0.962     |
| 3000      | 989               | 3,033,367      | 1,726.22 | 0.959     |
| 4000      | 1316              | 3,039,514      | 1,729.72 | 0.961     |

### Calculation Notes
- **Dhrystones/sec** = Iterations / (Execution Time / 1,000,000)
- **DMIPS** = Dhrystones/sec / 1757 (Baseline value)
- **DMIPS/MHz** = DMIPS / 1800 (Processor frequency)

### Performance Analysis

**Average Performance:**
- **Average Dhrystones/sec**: 3,036,831
- **Average DMIPS**: 1,728.17
- **Average DMIPS/MHz**: 0.960

---

## A5 Platform Test Results (Ascend910_9599 @ 1650MHz)

### Test Environment
- **Processor**: Ascend910_9599
- **Frequency**: 1650 MHz

### Test Data

| Iterations | Execution Time (μs) | Dhrystones/sec | DMIPS | DMIPS/MHz |
|-----------|-------------------|----------------|-------|-----------|
| 1000      | 375               | 2,666,667      | 1,517.25 | 0.920     |
| 2000      | 719               | 2,781,641      | 1,582.53 | 0.959     |
| 3000      | 1087              | 2,760,258      | 1,570.88 | 0.952     |
| 4000      | 1437              | 2,783,577      | 1,584.16 | 0.960     |

### Calculation Notes
- **Dhrystones/sec** = Iterations / (Execution Time / 1,000,000)
- **DMIPS** = Dhrystones/sec / 1757 (Baseline value)
- **DMIPS/MHz** = DMIPS / 1650 (Processor frequency)

### Performance Analysis

**Average Performance:**
- **Average Dhrystones/sec**: 2,748,036
- **Average DMIPS**: 1,563.71
- **Average DMIPS/MHz**: 0.948
