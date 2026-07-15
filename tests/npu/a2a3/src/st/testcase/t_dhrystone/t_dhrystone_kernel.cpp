/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.

The code snippet comes from zpu.

Copyright Zylin AS. All rights reserved.

The Dhrystone benchmark code is based on:
"Dhystone" Benchmark Program, C Version 2.1
Author: Reinhold P. Weicker, Siemens AG (1984, 1988)
Published in "Communications of the ACM" vol. 27, no. 10 (Oct. 1984)
This is a freely distributable industry-standard benchmark.
*/
#include "acl/acl.h"
#include "dhry.h"

using namespace pto;

AICORE void custom_strcpy(char* des, char* source)
{
    char* r = des;

    if ((des == NULL) || (source == NULL))
        return;

    while ((*r++ = *source++) != '\0')
        ;
}

AICORE int strcmp(char* sl, char* s2)
{
    for (; *sl == *s2; ++sl, ++s2)
        if (*sl == '\0')
            return (0);
    return ((*(unsigned char*)sl < *(unsigned char*)s2) ? -1 : +1);
}

AICORE void proc_1(Rec_Pointer ptr_val_par)
{
    Rec_Pointer ptr_glob = (Rec_Pointer)0x100000;
    Rec_Pointer next_record = ptr_val_par->ptr_comp;
    /* == ptr_glob_Next */
    /* Local variable, initialized with ptr_val_par->ptr_comp,    */
    /* corresponds to "rename" in Ada, "with" in Pascal           */

    structassign(*ptr_val_par->ptr_comp, *ptr_glob);
    ptr_val_par->variant.var_1.int_comp = 5;
    next_record->variant.var_1.int_comp = ptr_val_par->variant.var_1.int_comp;
    next_record->ptr_comp = ptr_val_par->ptr_comp;
    proc_3(&next_record->ptr_comp);
    /* ptr_val_par->ptr_comp->ptr_comp
                        == ptr_glob->ptr_comp */
    if (next_record->discr == Ident_1)
    /* then, executed */
    {
        next_record->variant.var_1.int_comp = 6;
        proc_6(ptr_val_par->variant.var_1.enum_comp, &next_record->variant.var_1.enum_comp);
        next_record->ptr_comp = ptr_glob->ptr_comp;
        proc_7(next_record->variant.var_1.int_comp, 10, &next_record->variant.var_1.int_comp);
    } else /* not executed */
        structassign(*ptr_val_par, *ptr_val_par->ptr_comp);
} /* proc_1 */

AICORE void proc_2(int* int_par_ref)

{
    char* ch_1_glob = (char*)0x100110;
    char* int_glob = (char*)0x100100;
    int int_loc;
    Enumeration enum_Loc;

    int_loc = *int_par_ref + 10;
    do /* executed once */
        if (*ch_1_glob == 'A')
        /* then, executed */
        {
            int_loc -= 1;
            *int_par_ref = int_loc - *int_glob;
            enum_Loc = Ident_1;
        }
    while (enum_Loc != Ident_1); /* true */
} /* proc_2 */

AICORE void proc_3(Rec_Pointer* ptr_ref_par)

{
    Rec_Pointer ptr_glob = (Rec_Pointer)0x100000;
    int* int_glob = (int*)0x100100;
    if (ptr_glob != NULL)
        /* then, executed */
        *ptr_ref_par = ptr_glob->ptr_comp;
    proc_7(10, *int_glob, &ptr_glob->variant.var_1.int_comp);
} /* proc_3 */

AICORE void proc_4(void)
{
    bool bool_Loc;

    bool* bool_glob = (bool*)0x100108;
    char* ch_1_glob = (char*)0x100110;
    char* ch_2_glob = (char*)0x100118;

    bool_Loc = *ch_1_glob == 'A';
    *bool_glob = bool_Loc | *bool_glob;
    *ch_2_glob = 'B';
} /* proc_4 */

AICORE void proc_5(void)
{
    bool* bool_glob = (bool*)0x100108;
    char* ch_1_glob = (char*)0x100110;

    *ch_1_glob = 'A';
    *bool_glob = false;
} /* proc_5 */

AICORE void proc_6(Enumeration enum_val_par, Enumeration* enum_ref_par)
{
    char* int_glob = (char*)0x100100;
    *enum_ref_par = enum_val_par;
    if (!func_3(enum_val_par))
        /* then, not executed */
        *enum_ref_par = Ident_4;
    switch (enum_val_par) {
        case Ident_1:
            *enum_ref_par = Ident_1;
            break;
        case Ident_2:
            if (*int_glob > 100)
                /* then */
                *enum_ref_par = Ident_1;
            else
                *enum_ref_par = Ident_4;
            break;
        case Ident_3: /* executed */
            *enum_ref_par = Ident_2;
            break;
        case Ident_4:
            break;
        case Ident_5:
            *enum_ref_par = Ident_3;
            break;
    } /* switch */
} /* proc_6 */

AICORE void proc_7(int int_1_par_val, int int_2_par_val, int* int_par_ref)
{
    int int_loc;

    int_loc = int_1_par_val + 2;
    *int_par_ref = int_2_par_val + int_loc;
} /* proc_7 */

AICORE void proc_8(int* arr_1_par_ref, int* arr_2_par_ref, int int_1_par_val, int int_2_par_val)
{
    int int_index;
    int int_loc;
    char* int_glob = (char*)0x100100;

    int_loc = int_1_par_val + 5;
    *(arr_1_par_ref + int_loc) = int_2_par_val;
    *(arr_1_par_ref + int_loc + 1) = *(arr_1_par_ref + int_loc);
    *(arr_1_par_ref + int_loc + 30) = int_loc;
    for (int_index = int_loc; int_index <= int_loc + 1; ++int_index)
        *(arr_2_par_ref + int_loc * 50 + int_index) = int_loc;
    (*(arr_2_par_ref + int_loc * 50 + int_loc - 1)) += 1;
    *(arr_2_par_ref + (int_loc + 20) * 50 + int_loc) = *(arr_1_par_ref + int_loc);
    *int_glob = 5;
} /* proc_8 */

AICORE Enumeration func_1(char ch_1_par_val, char ch_2_par_val)
/*************************************************/
/* executed three times                                         */
/* first call:      ch_1_par_val == 'H', ch_2_par_val == 'R'    */
/* second call:     ch_1_par_val == 'A', ch_2_par_val == 'C'    */
/* third call:      ch_1_par_val == 'B', ch_2_par_val == 'C'    */
{
    char ch_1_loc;
    char ch_2_loc;
    char* ch_1_glob = (char*)0x100110;

    ch_1_loc = ch_1_par_val;
    ch_2_loc = ch_1_loc;
    if (ch_2_loc != ch_2_par_val)
        /* then, executed */
        return (Ident_1);
    else /* not executed */
    {
        *ch_1_glob = ch_1_loc;
        return (Ident_2);
    }
} /* func_1 */

AICORE bool func_2(Str_30 str_1_par_ref, Str_30 str_2_par_ref)
/*************************************************/
/* executed once */
/* str_1_par_ref == "DHRYSTONE PROGRAM, 1'ST STRING" */
/* str_2_par_ref == "DHRYSTONE PROGRAM, 2'ND STRING" */
{
    int int_loc;
    char ch_loc;
    int* int_glob = (int*)0x100100;

    int_loc = 2;
    while (int_loc <= 2) /* loop body executed once */
        if (func_1(str_1_par_ref[int_loc], str_2_par_ref[int_loc + 1]) == Ident_1)
        /* then, executed */
        {
            ch_loc = 'A';
            int_loc += 1;
        } /* if, while */
    if (ch_loc >= 'W' && ch_loc < 'Z')
        /* then, not executed */
        int_loc = 7;
    if (ch_loc == 'R')
        /* then, not executed */
        return true;
    else /* executed */
    {
        if (strcmp(str_1_par_ref, str_2_par_ref) > 0)
        /* then, not executed */
        {
            int_loc += 7;
            *int_glob = int_loc;
            return true;
        } else /* executed */
            return false;
    } /* if ch_loc */
} /* func_2 */

AICORE bool func_3(Enumeration enum_par_val)
/***************************/
/* executed once        */
/* enum_par_val == Ident_3 */
{
    Enumeration enum_Loc;

    enum_Loc = enum_par_val;
    if (enum_Loc == Ident_3)
        /* then, executed */
        return true;
    else /* not executed */
        return false;
} /* func_3 */

template <int iteration>
__global__ AICORE void runTDhrystone()
{
    int int_1_Loc;
    int int_2_Loc;
    int int_3_Loc;
    char ch_Index;
    Enumeration enum_Loc;
    Str_30 str_1_loc;
    Str_30 str_2_loc;
    int run_index;
    int number_of_runs;

    Rec_Pointer ptr_glob = (Rec_Pointer)0x100000;
    Rec_Pointer next_ptr_glob = (Rec_Pointer)0x100080;
    int* int_glob = (int*)0x100100;
    bool* bool_glob = (bool*)0x100108;
    char* ch_2_glob = (char*)0x100118;
    int* arr_1_Glob = (int*)0x101000;
    int* arr_2_Glob = (int*)0x102000;

    /* Initializations */
    char cpystr0[31] = {'D', 'H', 'R', 'Y', 'S', 'T', 'O', 'N', 'E', ' ', 'P', 'R', 'O', 'G', 'R',
                        'A', 'M', ',', ' ', 'S', 'O', 'M', 'E', ' ', 'S', 'T', 'R', 'I', 'N', 'G'};
    char cpystr1[31] = {'D', 'H', 'R', 'Y', 'S', 'T',        'O', 'N', 'E', ' ', 'P', 'R', 'O', 'G',
                        'R', 'A', 'M', ',', '1', (char)0xde, 'S', 'T', 'S', 'T', 'R', 'I', 'N', 'G'};
    char cpystr2[31] = {'D', 'H', 'R', 'Y', 'S', 'T',        'O', 'N', 'E', ' ', 'P', 'R', 'O', 'G',
                        'R', 'A', 'M', ',', '2', (char)0xde, 'N', 'D', 'S', 'T', 'R', 'I', 'N', 'G'};
    char cpystr3[31] = {'D', 'H', 'R', 'Y', 'S', 'T',        'O', 'N', 'E', ' ', 'P', 'R', 'O', 'G',
                        'R', 'A', 'M', ',', '3', (char)0xde, 'R', 'D', 'S', 'T', 'R', 'I', 'N', 'G'};

    ptr_glob->ptr_comp = next_ptr_glob;
    ptr_glob->discr = Ident_1;
    ptr_glob->variant.var_1.enum_comp = Ident_3;
    ptr_glob->variant.var_1.int_comp = 40;
    custom_strcpy(ptr_glob->variant.var_1.str_comp, cpystr0);
    custom_strcpy(str_1_loc, cpystr1);

    *(arr_2_Glob + 8 * 50 + 7) = 10;

    number_of_runs = iteration;

#ifdef _DEBUG
    cce::printf("Execution starts, %d runs through Dhrystone\n", number_of_runs);
#endif

    /***************/
    /* Start timer */
    /***************/
    uint64_t tStart = get_sys_cnt();

    for (run_index = 1; run_index <= number_of_runs; ++run_index) {
        proc_5();
        proc_4();
        /* ch_1_glob == 'A', ch_2_glob == 'B', bool_glob == true */
        int_1_Loc = 2;
        int_2_Loc = 3;
        custom_strcpy(str_2_loc, cpystr2);
        enum_Loc = Ident_2;
        *bool_glob = !func_2(str_1_loc, str_2_loc);
        /* bool_glob == 1 */
        while (int_1_Loc < int_2_Loc) /* loop body executed once */
        {
            int_3_Loc = 5 * int_1_Loc - int_2_Loc;
            /* int_3_Loc == 7 */
            proc_7(int_1_Loc, int_2_Loc, &int_3_Loc);
            /* int_3_Loc == 7 */
            int_1_Loc += 1;
        } /* while */
        /* int_1_Loc == 3, int_2_Loc == 3, int_3_Loc == 7 */
        proc_8(arr_1_Glob, arr_2_Glob, int_1_Loc, int_3_Loc);
        /* int_glob == 5 */
        proc_1(ptr_glob);
        for (ch_Index = 'A'; ch_Index <= *ch_2_glob; ++ch_Index)
        /* loop body executed twice */
        {
            if (enum_Loc == func_1(ch_Index, 'C'))
            /* then, not executed */
            {
                proc_6(Ident_1, &enum_Loc);
                custom_strcpy(str_2_loc, cpystr3);
                int_2_Loc = run_index;
                *int_glob = run_index;
            }
        }
        /* int_1_Loc == 3, int_2_Loc == 3, int_3_Loc == 7 */
        int_2_Loc = int_2_Loc * int_1_Loc;
        int_1_Loc = int_2_Loc / int_3_Loc;
        int_2_Loc = 7 * (int_2_Loc - int_3_Loc) - int_1_Loc;
        /* int_1_Loc == 1, int_2_Loc == 13, int_3_Loc == 7 */
        proc_2(&int_1_Loc);
        /* int_1_Loc == 5 */
    } /* loop "for run_index" */

    /**************/
    /* Stop timer */
    /**************/

    pipe_barrier(PIPE_ALL);
    uint64_t tEnd = get_sys_cnt();

#ifdef _DEBUG
    cce::printf("Start @%d End @%d (%d us)\n", int(tStart), int(tEnd), int(tEnd - tStart) * 20 / 1000);
#endif
}

__global__ AICORE __attribute__((aic)) void warmup_kernel() {}

template <int iteration>
void LaunchTDhrystone(void* stream)
{
    warmup_kernel<<<24, nullptr, stream>>>();
    runTDhrystone<iteration><<<1, nullptr, stream>>>();
}

template void LaunchTDhrystone<1000>(void* stream);
template void LaunchTDhrystone<2000>(void* stream);
template void LaunchTDhrystone<3000>(void* stream);
template void LaunchTDhrystone<4000>(void* stream);
template void LaunchTDhrystone<5000>(void* stream);
template void LaunchTDhrystone<6000>(void* stream);
template void LaunchTDhrystone<7000>(void* stream);
template void LaunchTDhrystone<8000>(void* stream);