/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef HIFLOAT8_HPP
#define HIFLOAT8_HPP

#include <iostream>
#include <array>
#include <cmath>
#include <cstdint>
#include <algorithm>

class Hifloat8LUT {
private:
    static constexpr int tableSize = 256;
    static constexpr int denormalPrefixShift = 3;
    static constexpr double two = 2.0;
    static constexpr int prefix2BitShift = 5;
    static constexpr int prefix3BitShift = 4;
    static constexpr int prefix4BitShift = 3;
    static constexpr int exponentBitsFor2BitPrefix3 = 4;
    static constexpr int exponentBitsFor2BitPrefix2 = 3;
    static constexpr int exponentBitsFor2BitPrefix1 = 2;
    static constexpr int exponentBitsFor3BitPrefix001 = 1;
    static constexpr int exponentBitsFor4BitPrefix1 = 0;
    static constexpr int totalExponentMantissaBits = 5;
    static constexpr double infinityExponent = 15.0;
    static constexpr int prefixThree = 3;
    static constexpr int prefixTwo = 2;
    static constexpr int prefixOne = 1;
    static constexpr double denormalExponentOffset = 23.0;

public:
    std::array<double, tableSize> toDoubleTable;
    std::array<uint8_t, tableSize> sortedIndices;

    Hifloat8LUT()
    {
        for (int i = 0; i < tableSize; ++i) {
            toDoubleTable[i] = decodeHiF8Mathematically(static_cast<uint8_t>(i));
            sortedIndices[i] = static_cast<uint8_t>(i);
        }

        std::sort(sortedIndices.begin(), sortedIndices.end(), [this](uint8_t a, uint8_t b) {
            bool aIsNaN = std::isnan(toDoubleTable[a]);
            bool bIsNaN = std::isnan(toDoubleTable[b]);
            if (std::isnan(toDoubleTable[a]) && std::isnan(toDoubleTable[b]))
                return false;
            if (std::isnan(toDoubleTable[a]))
                return false;
            if (std::isnan(toDoubleTable[b]))
                return true;
            return toDoubleTable[a] < toDoubleTable[b];
        });
    }

private:
    double decodeHiF8Mathematically(uint8_t byte)
    {
        double sign = (byte & 0x80) ? -1.0 : 1.0;
        uint8_t payload = byte & 0x7F; // 7 bits left for Dot, Exponent, Mantissa

        // --- DENORMAL MODE (DML) / ZERO / NaN ---
        // Pattern 0000xxxx
        if ((payload >> denormalPrefixShift) == 0) {
            uint8_t mantissa = payload & 0x07; // 3 bits of mantissa
            if (mantissa == 0) {
                return (byte & 0x80) ? std::numeric_limits<double>::quiet_NaN() : 0.0;
            }
            // Equation (2): X = (-1)^S * 2^(M - 23) * 1.0
            return sign * std::pow(two, static_cast<double>(mantissa) - denormalExponentOffset);
        }

        // --- NORMAL MODES (NML) ---
        int exponentBits = 0;
        int remainingBits = 0;

        // Check 2-bit prefixes (examine top 2 bits of prefix4bits)
        if ((payload >> prefix2BitShift) == prefixThree) {
            exponentBits = exponentBitsFor2BitPrefix3;
            remainingBits = payload & 0x1F;
        } else if ((payload >> prefix2BitShift) == prefixTwo) {
            exponentBits = exponentBitsFor2BitPrefix2;
            remainingBits = payload & 0x1F;
        } else if ((payload >> prefix2BitShift) == prefixOne) {
            exponentBits = exponentBitsFor2BitPrefix1;
            remainingBits = payload & 0x1F;
        }
        // Check 3-bit prefix (examine top 3 bits of prefix4bits)
        else if ((payload >> prefix3BitShift) == prefixOne) {
            exponentBits = exponentBitsFor3BitPrefix001;
            remainingBits = payload & 0x0F; // 4 remaining bits
        }
        // Check 4-bit prefix
        else if ((payload >> prefix4BitShift) == prefixOne) {
            exponentBits = exponentBitsFor4BitPrefix1;
            remainingBits = payload & 0x07; // 3 remaining bits
        }

        int mantissaBits = totalExponentMantissaBits - exponentBits;
        int rawExponent = remainingBits >> mantissaBits;
        int rawMantissa = remainingBits & ((1 << mantissaBits) - 1);

        // Calculate Exponent using Sign-Magnitude with an implicit leading 1
        double actualExponent = 0.0;
        if (exponentBits > 0) {
            uint8_t expSignBit = rawExponent & (1 << (exponentBits - 1));
            uint8_t expMagnitudeBits = rawExponent - expSignBit;

            // The absolute magnitude has a fixed implicit MSB equal to 1
            double expMagnitude = static_cast<double>((1 << (exponentBits - 1)) | expMagnitudeBits);
            actualExponent = expSignBit ? -expMagnitude : expMagnitude;
        }

        // Handle Infinities mapping (Largest absolute values at D=4, where E=15 and M=1)
        if (exponentBits == exponentBitsFor2BitPrefix3 && actualExponent == infinityExponent && rawMantissa == 1) {
            return sign * std::numeric_limits<double>::infinity();
        }

        // Reconstruct Fractional Mantissa: 1.M
        double actualMantissa = 1.0 + (static_cast<double>(rawMantissa) / (1 << mantissaBits));

        // Equation (1): X = (-1)^S * 2^E * 1.M
        return sign * std::pow(two, actualExponent) * actualMantissa;
    }
};

struct hifloat8_t {
    uint8_t data;

    hifloat8_t() : data(0) {}

    static inline hifloat8_t FromRaw(uint8_t rawData) { return hifloat8_t(rawData, true); }

    hifloat8_t(double value)
    {
        if (std::isnan(value)) {
            data = 0x80;
            return;
        }

        auto it = std::lower_bound(
            hf8LUT.sortedIndices.begin(), hf8LUT.sortedIndices.end(), value, [](uint8_t index, double val) {
                if (std::isnan(hf8LUT.toDoubleTable[index]))
                    return false;
                return hf8LUT.toDoubleTable[index] < val;
            });

        if (it == hf8LUT.sortedIndices.end() || std::isnan(hf8LUT.toDoubleTable[*it])) {
            auto validIt = hf8LUT.sortedIndices.end() - 1;
            while (validIt != hf8LUT.sortedIndices.begin() && std::isnan(hf8LUT.toDoubleTable[*validIt])) {
                --validIt;
            }
            data = *validIt;
            return;
        }
        if (it == hf8LUT.sortedIndices.begin()) {
            data = *hf8LUT.sortedIndices.begin();
            return;
        }

        double diffCurrent = std::abs(hf8LUT.toDoubleTable[*it] - value);
        double diffPrevious = std::abs(hf8LUT.toDoubleTable[*(it - 1)] - value);

        data = (diffCurrent < diffPrevious) ? *it : *(it - 1);
    }

    uint8_t RawData() const { return data; }

    friend std::ostream& operator<<(std::ostream& stream, const hifloat8_t& value)
    {
        return stream << static_cast<double>(value);
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

    operator double() const { return hf8LUT.toDoubleTable[data]; }

protected:
    const inline static Hifloat8LUT hf8LUT;
    explicit hifloat8_t(uint8_t rawValue, bool dummy) : data(rawValue) {}
};

#endif // HIFLOAT8_HPP
