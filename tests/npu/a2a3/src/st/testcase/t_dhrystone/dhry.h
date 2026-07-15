/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.

The Dhrystone benchmark code is based on:
"Dhystone" Benchmark Program, C Version 2.1
Author: Reinhold P. Weicker, Siemens AG (1984, 1988)
Published in "Communications of the ACM" vol. 27, no. 10 (Oct. 1984)
This is a freely distributable industry-standard benchmark.
*/
#include <stdio.h>
#include <pto/common/constants.hpp>

#ifdef NOSTRUCTASSIGN
#define structassign(d, s) memcpy(&(d), &(s), sizeof(d))
#else
#define structassign(d, s) d = s
#endif

enum Enumeration { Ident_1, Ident_2, Ident_3, Ident_4, Ident_5 };

typedef char Str_30[31];

typedef struct record {
    struct record* ptr_comp;
    Enumeration discr;
    union {
        struct {
            Enumeration enum_comp;
            int int_comp;
            char str_comp[31];
        } var_1;
    } variant;
} Rec_Type, *Rec_Pointer;

AICORE void custom_strcpy(char* des, char* source);
AICORE int strcmp(char* sl, char* s2);

AICORE void proc_1(Rec_Pointer ptr_val_par);
AICORE void proc_2(int* int_par_ref);
AICORE void proc_3(Rec_Pointer* ptr_ref_par);
AICORE void proc_4(void);
AICORE void proc_5(void);
AICORE void proc_6(Enumeration enum_val_par, Enumeration* enum_ref_par);
AICORE void proc_7(int int_1_par_val, int int_2_par_val, int* int_par_ref);
AICORE void proc_8(int* arr_1_par_ref, int* arr_2_par_ref, int int_1_par_val, int int_2_par_val);

AICORE Enumeration func_1(char ch_1_par_val, char ch_2_par_val);
AICORE bool func_2(Str_30 str_1_par_ref, Str_30 str_2_par_ref);
AICORE bool func_3(Enumeration enum_par_val);
