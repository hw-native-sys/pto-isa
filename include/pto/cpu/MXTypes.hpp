/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MXTYPES_HPP
#define MXTYPES_HPP

constexpr unsigned int MAN_DBL = 52;
constexpr unsigned int EXP_DBL = 11;
constexpr int EXP_DBL_BIAS = 1023;
constexpr int RESERVED_EXPONENT_COUNT = 2;

template <int EXP_SZ, int MAN_SZ, int EXP_BIAS, bool IS_X2>
class MXType {
public:
    MXType() : data(0) {}

    static inline MXType FromRaw(uint8_t rawData) { return MXType(rawData, true); }

    MXType(double value)
    {
        uint64_t data = 0;

        // Handle zero explicitly
        if (value == 0.0) {
            data = 0;
            return;
        }

        // Extract bits from the double
        uint64_t& dblBits = *((uint64_t*)&value);

        uint64_t dblSign = (dblBits >> (MAN_DBL + EXP_DBL)) & 1;
        int64_t dblExponent = ((dblBits >> MAN_DBL) & ((1ULL << EXP_DBL) - 1));
        uint64_t dblMantissa = dblBits & ((1ULL << MAN_DBL) - 1);
        int64_t outExponent = 0;
        uint64_t outMantissa = 0;

        // Out-of-bounds values
        if (dblExponent - EXP_DBL_BIAS > (1ULL << EXP_SZ) - RESERVED_EXPONENT_COUNT) {
            // MIN, MAX, INF
            if (((dblBits >> MAN_DBL) & ((1ULL << EXP_DBL) - 1)) == (1ULL << EXP_DBL) - 1) {
                outExponent = (1ULL << EXP_SZ) - 1;
                outMantissa = dblMantissa & 1;
            }
        }

        // Normalized number
        if (dblExponent > 0) {
            // Scale mantissa to fit into custom size
            outMantissa = dblMantissa >> (MAN_DBL - MAN_SZ);

            // Adjust for custom bias
            outExponent = dblExponent + EXP_BIAS - EXP_DBL_BIAS;
            if (outExponent < 0) {
                outMantissa = (outMantissa | (1 << MAN_SZ)) >> (1 - outExponent);
                outExponent = 0;
                return;
            }
        } else {
            // Subnormals of double are too small to fit into new number
            // So, treat as zero
            data = 0;
            return;
        }

        data = (dblSign << (MAN_SZ + EXP_SZ)) | (outExponent << MAN_SZ) | outMantissa;
    }

    operator double() const
    {
        uint64_t mantissa = (data & ((1 << MAN_SZ) - 1));
        uint64_t exponent = (data >> MAN_SZ) & ((1 << EXP_SZ) - 1);
        uint64_t sign = (data >> (MAN_SZ + EXP_SZ)) & 1;

        double retVal = 0;

        if (exponent > 0 || MAN_SZ == 0) {
            // Normal representation
            *((uint64_t*)&retVal) = (sign << (MAN_DBL + EXP_DBL)) | ((exponent + EXP_DBL_BIAS - EXP_BIAS) << MAN_DBL) |
                                    (mantissa << (MAN_DBL - MAN_SZ));
        } else if (data == 0) {
            retVal = 0;
        } else if (data == 1ULL << (MAN_SZ + EXP_SZ)) {
            retVal = -0;
        } else {
            int i = MAN_SZ - 1; // Idx of the first mantissa bit equal to 1
            for (; i >= 0 && !((mantissa >> i) & 1); i--)
                ;
            // Subnormal representation
            *((uint64_t*)&retVal) = (sign << (MAN_DBL + EXP_DBL)) |
                                    (((uint64_t)(EXP_DBL_BIAS - EXP_BIAS + 1 + (i - MAN_SZ))) << MAN_DBL) |
                                    ((mantissa & ((1 << i) - 1)) << (MAN_DBL - i));
        }
        return retVal;
    }

    uint8_t RawData() const { return data; }

    friend std::ostream& operator<<(std::ostream& stream, const MXType<EXP_SZ, MAN_SZ, EXP_BIAS, IS_X2>& value)
    {
        return stream << static_cast<double>(value);
        // << std::hex << "(Raw: " << (uint32_t)value.RawData() << std::dec << ")";
    }

    template <typename SECOND_TYPE>
    double operator*(const SECOND_TYPE& op2) const
    {
        return static_cast<double>(*this) * static_cast<double>(op2);
    }

    template <typename SECOND_TYPE>
    double operator/(const SECOND_TYPE& op2) const
    {
        return static_cast<double>(*this) / static_cast<double>(op2);
    }

    template <typename SECOND_TYPE>
    double operator+(const SECOND_TYPE& op2) const
    {
        return static_cast<double>(*this) + static_cast<double>(op2);
    }

    template <typename SECOND_TYPE>
    double operator-(const SECOND_TYPE& op2) const
    {
        return static_cast<double>(*this) - static_cast<double>(op2);
    }

protected:
    explicit MXType(uint8_t rawValue, bool dummy) : data(rawValue) {}

    uint8_t data;
};

// ========== Constants for MXType template parameters ==========

// float4_e2m1x2_t: exponent=2 bits, mantissa=1 bit, bias=1
constexpr int kFloat4E2M1ExponentBits = 2;
constexpr int kFloat4E2M1MantissaBits = 1;
constexpr int kFloat4E2M1Bias = 1;

// float4_e1m2x2_t: exponent=1 bit, mantissa=2 bits, bias=1
constexpr int kFloat4E1M2ExponentBits = 1;
constexpr int kFloat4E1M2MantissaBits = 2;
constexpr int kFloat4E1M2Bias = 1;

// float8_e8m0_t: exponent=8 bits, mantissa=0 bits, bias=127
constexpr int kFloat8E8M0ExponentBits = 8;
constexpr int kFloat8E8M0MantissaBits = 0;
constexpr int kFloat8E8M0Bias = 127;

// float8_e4m3_t: exponent=4 bits, mantissa=3 bits, bias=7
constexpr int kFloat8E4M3ExponentBits = 4;
constexpr int kFloat8E4M3MantissaBits = 3;
constexpr int kFloat8E4M3Bias = 7;

// float8_e5m2_t: exponent=5 bits, mantissa=2 bits, bias=15
constexpr int kFloat8E5M2ExponentBits = 5;
constexpr int kFloat8E5M2MantissaBits = 2;
constexpr int kFloat8E5M2Bias = 15;

// Using declarations (can remain as is, or use the constants as shown below)
using float4_e2m1x2_t = MXType<kFloat4E2M1ExponentBits, kFloat4E2M1MantissaBits, kFloat4E2M1Bias, true>;
using float4_e1m2x2_t = MXType<kFloat4E1M2ExponentBits, kFloat4E1M2MantissaBits, kFloat4E1M2Bias, true>;
using float8_e8m0_t = MXType<kFloat8E8M0ExponentBits, kFloat8E8M0MantissaBits, kFloat8E8M0Bias, false>;
using float8_e4m3_t = MXType<kFloat8E4M3ExponentBits, kFloat8E4M3MantissaBits, kFloat8E4M3Bias, false>;
using float8_e5m2_t = MXType<kFloat8E5M2ExponentBits, kFloat8E5M2MantissaBits, kFloat8E5M2Bias, false>;

template <typename T>
constexpr bool isTwinType()
{
    return std::is_same_v<T, float4_e2m1x2_t> || std::is_same_v<T, float4_e1m2x2_t>;
}

#define HALF_BYTE_MASK 0xF
#define HALF_BYTE_SHIFT 4

template <typename T>
inline T getProperDataPart(T* buf, size_t offset)
{
    if constexpr (isTwinType<T>()) {
        // For FP4 data types we split byte into 2 parts at load operation and then operate with them as single bytes
        return T::FromRaw((buf[offset / 2].RawData() >> ((offset % 2) ? HALF_BYTE_SHIFT : 0)) & HALF_BYTE_MASK);
    } else {
        return buf[offset];
    }
}

template <typename T>
inline void setProperDataPart(T* buf, size_t offset, T val)
{
    if constexpr (isTwinType<T>()) {
        uint16_t shiftByte = (offset % 2) ? HALF_BYTE_SHIFT : 0;
        uint8_t rawVal = (val.RawData() & HALF_BYTE_MASK) << shiftByte;
        buf[offset / 2] = T::FromRaw((buf[offset / 2].RawData() & ~(HALF_BYTE_MASK << shiftByte)) | rawVal);
    } else {
        buf[offset] = val;
    }
}

#endif