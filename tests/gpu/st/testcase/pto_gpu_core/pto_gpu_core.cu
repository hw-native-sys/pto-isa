#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>

#include <pto/pto-inst.hpp>

namespace {

template <typename T, pto::Layout LayoutV>
struct SimpleGlobal {
    using DType = T;
    static constexpr pto::Layout layout = LayoutV;

    T *ptr;
    int64_t shape[pto::GlobalTensorDim::TOTAL_DIM];
    int64_t stride[pto::GlobalTensorDim::TOTAL_DIM];

    __host__ __device__ SimpleGlobal(T *ptr_, int64_t s0, int64_t s1, int64_t s2, int64_t s3, int64_t s4,
                                     int64_t st0, int64_t st1, int64_t st2, int64_t st3, int64_t st4)
        : ptr(ptr_), shape{s0, s1, s2, s3, s4}, stride{st0, st1, st2, st3, st4}
    {
    }

    __host__ __device__ T *data()
    {
        return ptr;
    }

    __host__ __device__ const T *data() const
    {
        return ptr;
    }

    __host__ __device__ int64_t GetShape(int dim) const
    {
        return shape[dim];
    }

    __host__ __device__ int64_t GetStride(int dim) const
    {
        return stride[dim];
    }
};

bool CheckCuda(cudaError_t status, const char *what)
{
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << "[CUDA] " << what << ": " << cudaGetErrorString(status) << std::endl;
    return false;
}

template <typename T>
std::size_t RowMajorOffset(std::size_t rows, std::size_t cols, std::size_t r, std::size_t c)
{
    (void)rows;
    return r * cols + c;
}

template <typename T>
std::size_t ColMajorOffset(std::size_t rows, std::size_t cols, std::size_t r, std::size_t c)
{
    (void)cols;
    return c * rows + r;
}

template <typename T>
std::vector<T> RefTLoadNdRowMajor(const std::vector<T> &src, int g0, int g1, int g2, int g3, int g4, int tileRows,
                                  int tileCols, T pad)
{
    std::vector<T> out(tileRows * tileCols, pad);
    int rowBase = 0;
    for (int i = 0; i < g0; ++i) {
        for (int j = 0; j < g1; ++j) {
            for (int k = 0; k < g2; ++k) {
                for (int r = 0; r < g3; ++r) {
                    const int tileRow = rowBase + r;
                    const int srcBase = i * (g1 * g2 * g3 * g4) + j * (g2 * g3 * g4) + k * (g3 * g4) + r * g4;
                    for (int c = 0; c < g4; ++c) {
                        out[RowMajorOffset<T>(tileRows, tileCols, tileRow, c)] = src[srcBase + c];
                    }
                }
                rowBase += g3;
            }
        }
    }
    return out;
}

template <typename T>
std::vector<T> RefTLoadDnColMajor(const std::vector<T> &src, int g0, int g1, int g2, int g3, int g4, int tileRows,
                                  int tileCols, T pad)
{
    std::vector<T> out(tileRows * tileCols, pad);
    int colBase = 0;
    for (int i = 0; i < g0; ++i) {
        for (int j = 0; j < g1; ++j) {
            for (int k = 0; k < g2; ++k) {
                for (int c = 0; c < g4; ++c) {
                    const int tileCol = colBase + c;
                    const int srcBase = i * (g1 * g2 * g3 * g4) + j * (g2 * g3 * g4) + k * (g3 * g4) + c * g3;
                    for (int r = 0; r < g3; ++r) {
                        out[ColMajorOffset<T>(tileRows, tileCols, r, tileCol)] = src[srcBase + r];
                    }
                }
                colBase += g4;
            }
        }
    }
    return out;
}

template <typename T>
std::vector<T> RefTStoreNdRowMajor(const std::vector<T> &tile, int g0, int g1, int g2, int g3, int g4, int tileRows,
                                   int tileCols)
{
    std::vector<T> out(g0 * g1 * g2 * g3 * g4, T{});
    int rowBase = 0;
    for (int i = 0; i < g0; ++i) {
        for (int j = 0; j < g1; ++j) {
            for (int k = 0; k < g2; ++k) {
                for (int r = 0; r < g3; ++r) {
                    const int tileRow = rowBase + r;
                    const int dstBase = i * (g1 * g2 * g3 * g4) + j * (g2 * g3 * g4) + k * (g3 * g4) + r * g4;
                    for (int c = 0; c < g4; ++c) {
                        out[dstBase + c] = tile[RowMajorOffset<T>(tileRows, tileCols, tileRow, c)];
                    }
                }
                rowBase += g3;
            }
        }
    }
    return out;
}

template <typename T>
std::vector<T> RefTStoreDnColMajor(const std::vector<T> &tile, int g0, int g1, int g2, int g3, int g4, int tileRows,
                                   int tileCols)
{
    std::vector<T> out(g0 * g1 * g2 * g3 * g4, T{});
    int colBase = 0;
    for (int i = 0; i < g0; ++i) {
        for (int j = 0; j < g1; ++j) {
            for (int k = 0; k < g2; ++k) {
                for (int c = 0; c < g4; ++c) {
                    const int tileCol = colBase + c;
                    const int dstBase = i * (g1 * g2 * g3 * g4) + j * (g2 * g3 * g4) + k * (g3 * g4) + c * g3;
                    for (int r = 0; r < g3; ++r) {
                        out[dstBase + r] = tile[ColMajorOffset<T>(tileRows, tileCols, r, tileCol)];
                    }
                }
                colBase += g4;
            }
        }
    }
    return out;
}

template <typename T>
std::vector<T> RefTAddRowMajor(const std::vector<T> &a, const std::vector<T> &b, int tileRows, int tileCols,
                               int validRows, int validCols, T sentinel)
{
    std::vector<T> out(tileRows * tileCols, sentinel);
    for (int r = 0; r < validRows; ++r) {
        for (int c = 0; c < validCols; ++c) {
            out[RowMajorOffset<T>(tileRows, tileCols, r, c)] =
                a[RowMajorOffset<T>(tileRows, tileCols, r, c)] + b[RowMajorOffset<T>(tileRows, tileCols, r, c)];
        }
    }
    return out;
}

template <typename T>
std::vector<T> RefGpuSwizzle128BRowMajor(const std::vector<T> &logical, int rows, int cols)
{
    constexpr int swizzleRows = 8;
    const int swizzleCols = 128 / (swizzleRows * sizeof(T));
    std::vector<T> out(logical.size(), T{});
    const int chunksPerRow = cols / swizzleCols;
    auto isPow2 = [](int v) { return v > 0 && ((v & (v - 1)) == 0); };
    for (int r = 0; r < rows; ++r) {
        const int rowBlock = r / swizzleRows;
        const int rowInBlock = r % swizzleRows;
        for (int c = 0; c < cols; ++c) {
            const int chunk = c / swizzleCols;
            const int colInChunk = c % swizzleCols;
            const int permutedChunk = isPow2(chunksPerRow) ? ((chunk ^ (rowInBlock % chunksPerRow)) & (chunksPerRow - 1))
                                                           : ((chunk + rowInBlock) % chunksPerRow);
            const int physical = rowBlock * swizzleRows * cols + rowInBlock * cols + permutedChunk * swizzleCols + colInChunk;
            out[physical] = logical[r * cols + c];
        }
    }
    return out;
}

template <typename T>
std::vector<T> RefMatmulF32(const std::vector<T> &a, const std::vector<T> &b, int m, int k, int n)
{
    std::vector<T> out(m * n, T{});
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            T acc = T{};
            for (int kk = 0; kk < k; ++kk) {
                acc += a[row * k + kk] * b[kk * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    return out;
}

template <typename T>
float ToHostFloat(T value)
{
    return static_cast<float>(value);
}

template <>
float ToHostFloat<half>(half value)
{
    return __half2float(value);
}

template <>
float ToHostFloat<bfloat16_t>(bfloat16_t value)
{
    return __bfloat162float(value);
}

template <typename T>
std::vector<float> RefMatmulToFloat(const std::vector<T> &a, const std::vector<T> &b, int m, int k, int n)
{
    std::vector<float> out(m * n, 0.0f);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                acc += ToHostFloat(a[row * k + kk]) * ToHostFloat(b[kk * n + col]);
            }
            out[row * n + col] = acc;
        }
    }
    return out;
}

template <typename T>
std::vector<float> RefMatmulAccToFloat(const std::vector<T> &a, const std::vector<T> &b, const std::vector<float> &acc,
                                       int m, int k, int n)
{
    auto out = RefMatmulToFloat(a, b, m, k, n);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] += acc[i];
    }
    return out;
}

template <typename T>
std::vector<float> RefMatmulBiasToFloat(const std::vector<T> &a, const std::vector<T> &b,
                                        const std::vector<float> &bias, int m, int k, int n)
{
    auto out = RefMatmulToFloat(a, b, m, k, n);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            out[row * n + col] += bias[col];
        }
    }
    return out;
}

template <typename T>
bool ExpectVecEq(const std::vector<T> &expected, const std::vector<T> &actual, const char *label)
{
    if (expected.size() != actual.size()) {
        std::cerr << label << ": size mismatch: expected " << expected.size() << ", got " << actual.size()
                  << std::endl;
        return false;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            std::cerr << label << ": mismatch at index " << i << std::endl;
            return false;
        }
    }
    return true;
}

template <>
bool ExpectVecEq<float>(const std::vector<float> &expected, const std::vector<float> &actual, const char *label)
{
    if (expected.size() != actual.size()) {
        std::cerr << label << ": size mismatch: expected " << expected.size() << ", got " << actual.size()
                  << std::endl;
        return false;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(expected[i] - actual[i]) > 1e-5f) {
            std::cerr << label << ": mismatch at index " << i << ", expected " << expected[i] << ", got "
                      << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

bool ExpectVecNear(const std::vector<float> &expected, const std::vector<float> &actual,
                   const char *label, float tol)
{
    if (expected.size() != actual.size()) {
        std::cerr << label << ": size mismatch: expected " << expected.size() << ", got " << actual.size()
                  << std::endl;
        return false;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::abs(expected[i] - actual[i]) > tol) {
            std::cerr << label << ": mismatch at index " << i << ", expected " << expected[i] << ", got "
                      << actual[i] << ", tol " << tol << std::endl;
            return false;
        }
    }
    return true;
}

template <typename T, int G0, int G1, int G2, int G3, int G4, int TRows, int TCols>
__global__ void KernelTLoadNd(T *out, T *src)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    using GlobalData = SimpleGlobal<T, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, TRows, TCols, pto::BLayout::RowMajor, -1, -1,
                               pto::SLayout::NoneBox, 512, pto::PadValue::Zero>;
    T storage[TRows * TCols];
    TileData tile(G0 * G1 * G2 * G3, G4);
    tile.data() = storage;
    GlobalData global(src, G0, G1, G2, G3, G4, G1 * G2 * G3 * G4, G2 * G3 * G4, G3 * G4, G4, 1);
    pto::TLOAD(tile, global);
    for (int i = 0; i < TRows * TCols; ++i) {
        out[i] = storage[i];
    }
}

template <typename T, int G0, int G1, int G2, int G3, int G4, int TRows, int TCols>
__global__ void KernelTLoadDn(T *out, T *src)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    using GlobalData = SimpleGlobal<T, pto::Layout::DN>;
    using TileData = pto::Tile<pto::TileType::Vec, T, TRows, TCols, pto::BLayout::ColMajor, -1, -1,
                               pto::SLayout::NoneBox, 512, pto::PadValue::Zero>;
    T storage[TRows * TCols];
    TileData tile(G3, G0 * G1 * G2 * G4);
    tile.data() = storage;
    GlobalData global(src, G0, G1, G2, G3, G4, G1 * G2 * G3 * G4, G2 * G3 * G4, G3 * G4, 1, G3);
    pto::TLOAD(tile, global);
    for (int i = 0; i < TRows * TCols; ++i) {
        out[i] = storage[i];
    }
}

template <typename T, int G0, int G1, int G2, int G3, int G4, int TRows, int TCols>
__global__ void KernelTStoreNd(T *out, T *tileRaw)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    using GlobalData = SimpleGlobal<T, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, T, TRows, TCols, pto::BLayout::RowMajor, -1, -1>;
    TileData tile(G0 * G1 * G2 * G3, G4);
    tile.data() = tileRaw;
    GlobalData global(out, G0, G1, G2, G3, G4, G1 * G2 * G3 * G4, G2 * G3 * G4, G3 * G4, G4, 1);
    pto::TSTORE(global, tile);
}

template <typename T, int G0, int G1, int G2, int G3, int G4, int TRows, int TCols>
__global__ void KernelTStoreDn(T *out, T *tileRaw)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    using GlobalData = SimpleGlobal<T, pto::Layout::DN>;
    using TileData = pto::Tile<pto::TileType::Vec, T, TRows, TCols, pto::BLayout::ColMajor, -1, -1>;
    TileData tile(G3, G0 * G1 * G2 * G4);
    tile.data() = tileRaw;
    GlobalData global(out, G0, G1, G2, G3, G4, G1 * G2 * G3 * G4, G2 * G3 * G4, G3 * G4, 1, G3);
    pto::TSTORE(global, tile);
}

template <typename T, int TRows, int TCols, int ValidRows, int ValidCols>
__global__ void KernelTAdd(T *out, T *a, T *b, T sentinel)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    using TileData = pto::Tile<pto::TileType::Vec, T, TRows, TCols, pto::BLayout::RowMajor, -1, -1>;
    T aStorage[TRows * TCols];
    T bStorage[TRows * TCols];
    T cStorage[TRows * TCols];
    for (int i = 0; i < TRows * TCols; ++i) {
        aStorage[i] = a[i];
        bStorage[i] = b[i];
        cStorage[i] = sentinel;
    }
    TileData aTile(ValidRows, ValidCols);
    TileData bTile(ValidRows, ValidCols);
    TileData cTile(ValidRows, ValidCols);
    aTile.data() = aStorage;
    bTile.data() = bStorage;
    cTile.data() = cStorage;
    pto::TADD(cTile, aTile, bTile);
    for (int i = 0; i < TRows * TCols; ++i) {
        out[i] = cStorage[i];
    }
}

template <typename T, int Rows, int Cols>
__global__ void KernelTLoadGpuSwizzleRaw(T *out, T *src)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    using Shape5 = pto::Shape<-1, -1, -1, -1, -1>;
    using GlobalData = SimpleGlobal<T, pto::Layout::ND>;
    using TileData = pto::TileVecGpuSwizzle<T, Rows, Cols>;
    T storage[Rows * Cols];
    TileData tile;
    tile.data() = storage;
    GlobalData global(src, 1, 1, 1, Rows, Cols, Rows * Cols, Rows * Cols, Rows * Cols, Cols, 1);
    pto::TLOAD(tile, global);
    for (int i = 0; i < Rows * Cols; ++i) {
        out[i] = storage[i];
    }
}

template <typename T, int Rows, int Cols>
__global__ void KernelTLoadStoreGpuSwizzle(T *out, T *src)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    using GlobalData = SimpleGlobal<T, pto::Layout::ND>;
    using TileData = pto::TileVecGpuSwizzle<T, Rows, Cols>;
    T storage[Rows * Cols];
    TileData tile;
    tile.data() = storage;
    GlobalData srcGlobal(src, 1, 1, 1, Rows, Cols, Rows * Cols, Rows * Cols, Rows * Cols, Cols, 1);
    GlobalData dstGlobal(out, 1, 1, 1, Rows, Cols, Rows * Cols, Rows * Cols, Rows * Cols, Cols, 1);
    pto::TLOAD(tile, srcGlobal);
    pto::TSTORE(dstGlobal, tile);
}

template <typename T, int Rows, int Cols>
__global__ void KernelTAddGpuSwizzleRoundTrip(T *out, T *a, T *b)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    using GlobalData = SimpleGlobal<T, pto::Layout::ND>;
    using TileData = pto::TileVecGpuSwizzle<T, Rows, Cols>;
    T aStorage[Rows * Cols];
    T bStorage[Rows * Cols];
    T cStorage[Rows * Cols];
    TileData aTile;
    TileData bTile;
    TileData cTile;
    aTile.data() = aStorage;
    bTile.data() = bStorage;
    cTile.data() = cStorage;
    GlobalData aGlobal(a, 1, 1, 1, Rows, Cols, Rows * Cols, Rows * Cols, Rows * Cols, Cols, 1);
    GlobalData bGlobal(b, 1, 1, 1, Rows, Cols, Rows * Cols, Rows * Cols, Rows * Cols, Cols, 1);
    GlobalData outGlobal(out, 1, 1, 1, Rows, Cols, Rows * Cols, Rows * Cols, Rows * Cols, Cols, 1);
    pto::TLOAD(aTile, aGlobal);
    pto::TLOAD(bTile, bGlobal);
    pto::TADD(cTile, aTile, bTile);
    pto::TSTORE(outGlobal, cTile);
}

template <typename InputT, int M, int K, int N>
__global__ void KernelTMATMUL(float *out, InputT *a, InputT *b)
{
    using TileA = pto::Tile<pto::TileType::Vec, InputT, M, K, pto::BLayout::RowMajor, -1, -1>;
    using TileB = pto::Tile<pto::TileType::Vec, InputT, K, N, pto::BLayout::RowMajor, -1, -1>;
    using TileC = pto::Tile<pto::TileType::Acc, float, M, N, pto::BLayout::RowMajor, -1, -1>;
    TileA aTile(M, K);
    TileB bTile(K, N);
    TileC cTile(M, N);
    aTile.data() = a;
    bTile.data() = b;
    cTile.data() = out;
    pto::TMATMUL(cTile, aTile, bTile);
}

template <typename InputT, int M, int K, int N>
__global__ void KernelTMATMUL_ACC(float *out, float *in, InputT *a, InputT *b)
{
    using TileA = pto::Tile<pto::TileType::Vec, InputT, M, K, pto::BLayout::RowMajor, -1, -1>;
    using TileB = pto::Tile<pto::TileType::Vec, InputT, K, N, pto::BLayout::RowMajor, -1, -1>;
    using TileC = pto::Tile<pto::TileType::Acc, float, M, N, pto::BLayout::RowMajor, -1, -1>;
    TileA aTile(M, K);
    TileB bTile(K, N);
    TileC cOutTile(M, N);
    TileC cInTile(M, N);
    aTile.data() = a;
    bTile.data() = b;
    cOutTile.data() = out;
    cInTile.data() = in;
    pto::TMATMUL_ACC(cOutTile, cInTile, aTile, bTile);
}

template <typename InputT, int M, int K, int N>
__global__ void KernelTMATMUL_BIAS(float *out, InputT *a, InputT *b, float *bias)
{
    using TileA = pto::Tile<pto::TileType::Vec, InputT, M, K, pto::BLayout::RowMajor, -1, -1>;
    using TileB = pto::Tile<pto::TileType::Vec, InputT, K, N, pto::BLayout::RowMajor, -1, -1>;
    using TileC = pto::Tile<pto::TileType::Acc, float, M, N, pto::BLayout::RowMajor, -1, -1>;
    using TileBias = pto::Tile<pto::TileType::Vec, float, 1, N, pto::BLayout::RowMajor, -1, -1>;
    TileA aTile(M, K);
    TileB bTile(K, N);
    TileC cTile(M, N);
    TileBias biasTile(1, N);
    aTile.data() = a;
    bTile.data() = b;
    cTile.data() = out;
    biasTile.data() = bias;
    pto::TMATMUL_BIAS(cTile, aTile, bTile, biasTile);
}

template <typename InputT, int M, int K, int N>
__global__ void KernelTMATMUL_MX(float *out, InputT *a, float *aScale, InputT *b, float *bScale)
{
    using TileA = pto::Tile<pto::TileType::Vec, InputT, M, K, pto::BLayout::RowMajor, -1, -1>;
    using TileB = pto::Tile<pto::TileType::Vec, InputT, K, N, pto::BLayout::RowMajor, -1, -1>;
    using TileC = pto::Tile<pto::TileType::Acc, float, M, N, pto::BLayout::RowMajor, -1, -1>;
    using TileScaleA = pto::Tile<pto::TileType::ScaleLeft, float, 1, K, pto::BLayout::RowMajor, -1, -1>;
    using TileScaleB = pto::Tile<pto::TileType::ScaleRight, float, 1, N, pto::BLayout::RowMajor, -1, -1>;
    TileA aTile(M, K);
    TileB bTile(K, N);
    TileC cTile(M, N);
    TileScaleA aScaleTile(1, K);
    TileScaleB bScaleTile(1, N);
    aTile.data() = a;
    bTile.data() = b;
    cTile.data() = out;
    aScaleTile.data() = aScale;
    bScaleTile.data() = bScale;
    pto::TMATMUL_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);
}

template <typename InputT, int M, int K, int N>
__global__ void KernelTGEMV_MX(float *out, InputT *a, float *aScale, InputT *b, float *bScale)
{
    using TileA = pto::Tile<pto::TileType::Vec, InputT, M, K, pto::BLayout::RowMajor, -1, -1>;
    using TileB = pto::Tile<pto::TileType::Vec, InputT, K, N, pto::BLayout::RowMajor, -1, -1>;
    using TileC = pto::Tile<pto::TileType::Acc, float, M, N, pto::BLayout::RowMajor, -1, -1>;
    using TileScaleA = pto::Tile<pto::TileType::ScaleLeft, float, 1, K, pto::BLayout::RowMajor, -1, -1>;
    using TileScaleB = pto::Tile<pto::TileType::ScaleRight, float, 1, N, pto::BLayout::RowMajor, -1, -1>;
    TileA aTile(M, K);
    TileB bTile(K, N);
    TileC cTile(M, N);
    TileScaleA aScaleTile(1, K);
    TileScaleB bScaleTile(1, N);
    aTile.data() = a;
    bTile.data() = b;
    cTile.data() = out;
    aScaleTile.data() = aScale;
    bScaleTile.data() = bScale;
    pto::TGEMV_MX(cTile, aTile, aScaleTile, bTile, bScaleTile);
}

bool TestTLoadNdRowMajorMatchesReference()
{
    constexpr int G0 = 1, G1 = 1, G2 = 1, G3 = 4, G4 = 7, TRows = 4, TCols = 8;
    std::vector<float> src(G0 * G1 * G2 * G3 * G4);
    for (std::size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<float>(i + 1);
    }
    auto expected = RefTLoadNdRowMajor(src, G0, G1, G2, G3, G4, TRows, TCols, 0.0f);

    float *dSrc = nullptr;
    float *dOut = nullptr;
    if (!CheckCuda(cudaMalloc(&dSrc, src.size() * sizeof(float)), "cudaMalloc dSrc")) return false;
    if (!CheckCuda(cudaMalloc(&dOut, expected.size() * sizeof(float)), "cudaMalloc dOut")) return false;
    if (!CheckCuda(cudaMemcpy(dSrc, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice), "copy src")) return false;
    KernelTLoadNd<float, G0, G1, G2, G3, G4, TRows, TCols><<<1, 1>>>(dOut, dSrc);
    if (!CheckCuda(cudaGetLastError(), "launch tload nd")) return false;
    if (!CheckCuda(cudaDeviceSynchronize(), "sync tload nd")) return false;

    std::vector<float> actual(expected.size());
    if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out")) return false;
    cudaFree(dSrc);
    cudaFree(dOut);
    return ExpectVecEq(expected, actual, "tload_nd");
}

bool TestTLoadDnColMajorMatchesReference()
{
    constexpr int G0 = 1, G1 = 1, G2 = 1, G3 = 3, G4 = 5, TRows = 8, TCols = 8;
    std::vector<float> src(G0 * G1 * G2 * G3 * G4);
    for (std::size_t i = 0; i < src.size(); ++i) {
        src[i] = static_cast<float>(100 + i);
    }
    auto expected = RefTLoadDnColMajor(src, G0, G1, G2, G3, G4, TRows, TCols, 0.0f);

    float *dSrc = nullptr;
    float *dOut = nullptr;
    if (!CheckCuda(cudaMalloc(&dSrc, src.size() * sizeof(float)), "cudaMalloc dSrc")) return false;
    if (!CheckCuda(cudaMalloc(&dOut, expected.size() * sizeof(float)), "cudaMalloc dOut")) return false;
    if (!CheckCuda(cudaMemcpy(dSrc, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice), "copy src")) return false;
    KernelTLoadDn<float, G0, G1, G2, G3, G4, TRows, TCols><<<1, 1>>>(dOut, dSrc);
    if (!CheckCuda(cudaGetLastError(), "launch tload dn")) return false;
    if (!CheckCuda(cudaDeviceSynchronize(), "sync tload dn")) return false;

    std::vector<float> actual(expected.size());
    if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out")) return false;
    cudaFree(dSrc);
    cudaFree(dOut);
    return ExpectVecEq(expected, actual, "tload_dn");
}

bool TestTStoreNdAndDnMatchReference()
{
    {
        constexpr int G0 = 1, G1 = 1, G2 = 1, G3 = 4, G4 = 7, TRows = 4, TCols = 8;
        std::vector<float> tile(TRows * TCols, -1.0f);
        for (int r = 0; r < G3; ++r) {
            for (int c = 0; c < G4; ++c) {
                tile[r * TCols + c] = static_cast<float>(r * 10 + c);
            }
        }
        auto expected = RefTStoreNdRowMajor(tile, G0, G1, G2, G3, G4, TRows, TCols);
        float *dTile = nullptr;
        float *dOut = nullptr;
        if (!CheckCuda(cudaMalloc(&dTile, tile.size() * sizeof(float)), "cudaMalloc dTile")) return false;
        if (!CheckCuda(cudaMalloc(&dOut, expected.size() * sizeof(float)), "cudaMalloc dOut")) return false;
        if (!CheckCuda(cudaMemset(dOut, 0, expected.size() * sizeof(float)), "memset dOut")) return false;
        if (!CheckCuda(cudaMemcpy(dTile, tile.data(), tile.size() * sizeof(float), cudaMemcpyHostToDevice), "copy tile")) return false;
        KernelTStoreNd<float, G0, G1, G2, G3, G4, TRows, TCols><<<1, 1>>>(dOut, dTile);
        if (!CheckCuda(cudaGetLastError(), "launch tstore nd")) return false;
        if (!CheckCuda(cudaDeviceSynchronize(), "sync tstore nd")) return false;
        std::vector<float> actual(expected.size());
        if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out")) return false;
        cudaFree(dTile);
        cudaFree(dOut);
        if (!ExpectVecEq(expected, actual, "tstore_nd")) return false;
    }
    {
        constexpr int G0 = 1, G1 = 1, G2 = 1, G3 = 3, G4 = 6, TRows = 16, TCols = 8;
        std::vector<int16_t> tile(TRows * TCols, static_cast<int16_t>(-7));
        for (int c = 0; c < G4; ++c) {
            for (int r = 0; r < G3; ++r) {
                tile[c * TRows + r] = static_cast<int16_t>(c * 10 + r);
            }
        }
        auto expected = RefTStoreDnColMajor(tile, G0, G1, G2, G3, G4, TRows, TCols);
        int16_t *dTile = nullptr;
        int16_t *dOut = nullptr;
        if (!CheckCuda(cudaMalloc(&dTile, tile.size() * sizeof(int16_t)), "cudaMalloc dTile")) return false;
        if (!CheckCuda(cudaMalloc(&dOut, expected.size() * sizeof(int16_t)), "cudaMalloc dOut")) return false;
        if (!CheckCuda(cudaMemset(dOut, 0, expected.size() * sizeof(int16_t)), "memset dOut")) return false;
        if (!CheckCuda(cudaMemcpy(dTile, tile.data(), tile.size() * sizeof(int16_t), cudaMemcpyHostToDevice), "copy tile")) return false;
        KernelTStoreDn<int16_t, G0, G1, G2, G3, G4, TRows, TCols><<<1, 1>>>(dOut, dTile);
        if (!CheckCuda(cudaGetLastError(), "launch tstore dn")) return false;
        if (!CheckCuda(cudaDeviceSynchronize(), "sync tstore dn")) return false;
        std::vector<int16_t> actual(expected.size());
        if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(int16_t), cudaMemcpyDeviceToHost), "copy out")) return false;
        cudaFree(dTile);
        cudaFree(dOut);
        if (!ExpectVecEq(expected, actual, "tstore_dn")) return false;
    }
    return true;
}

bool TestTAddMatchesReference()
{
    constexpr int TRows = 8, TCols = 8, ValidRows = 5, ValidCols = 7;
    std::vector<float> a(TRows * TCols);
    std::vector<float> b(TRows * TCols);
    for (int i = 0; i < TRows * TCols; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(100 - i);
    }
    constexpr float sentinel = -777.0f;
    auto expected = RefTAddRowMajor(a, b, TRows, TCols, ValidRows, ValidCols, sentinel);
    float *dA = nullptr;
    float *dB = nullptr;
    float *dOut = nullptr;
    if (!CheckCuda(cudaMalloc(&dA, a.size() * sizeof(float)), "cudaMalloc dA")) return false;
    if (!CheckCuda(cudaMalloc(&dB, b.size() * sizeof(float)), "cudaMalloc dB")) return false;
    if (!CheckCuda(cudaMalloc(&dOut, expected.size() * sizeof(float)), "cudaMalloc dOut")) return false;
    if (!CheckCuda(cudaMemcpy(dA, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice), "copy a")) return false;
    if (!CheckCuda(cudaMemcpy(dB, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice), "copy b")) return false;
    KernelTAdd<float, TRows, TCols, ValidRows, ValidCols><<<1, 1>>>(dOut, dA, dB, sentinel);
    if (!CheckCuda(cudaGetLastError(), "launch tadd")) return false;
    if (!CheckCuda(cudaDeviceSynchronize(), "sync tadd")) return false;
    std::vector<float> actual(expected.size());
    if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out")) return false;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dOut);
    return ExpectVecEq(expected, actual, "tadd");
}

bool TestGpuSwizzlePhysicalLayoutMatchesReference()
{
    constexpr int Rows = 16, Cols = 16;
    std::vector<half> src(Rows * Cols);
    for (int i = 0; i < Rows * Cols; ++i) {
        src[i] = __float2half(static_cast<float>(i + 1));
    }
    auto expected = RefGpuSwizzle128BRowMajor(src, Rows, Cols);
    half *dSrc = nullptr;
    half *dOut = nullptr;
    if (!CheckCuda(cudaMalloc(&dSrc, src.size() * sizeof(half)), "cudaMalloc dSrc swizzle")) return false;
    if (!CheckCuda(cudaMalloc(&dOut, expected.size() * sizeof(half)), "cudaMalloc dOut swizzle")) return false;
    if (!CheckCuda(cudaMemcpy(dSrc, src.data(), src.size() * sizeof(half), cudaMemcpyHostToDevice), "copy src swizzle")) return false;
    KernelTLoadGpuSwizzleRaw<half, Rows, Cols><<<1, 1>>>(dOut, dSrc);
    if (!CheckCuda(cudaGetLastError(), "launch tload swizzle raw")) return false;
    if (!CheckCuda(cudaDeviceSynchronize(), "sync tload swizzle raw")) return false;
    std::vector<half> actual(expected.size());
    if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(half), cudaMemcpyDeviceToHost), "copy out swizzle")) return false;
    cudaFree(dSrc);
    cudaFree(dOut);
    return ExpectVecEq(expected, actual, "gpu_swizzle_physical");
}

bool TestGpuSwizzleRoundTripMatchesReference()
{
    constexpr int Rows = 16, Cols = 16;
    std::vector<half> src(Rows * Cols);
    for (int i = 0; i < Rows * Cols; ++i) {
        src[i] = __float2half(static_cast<float>((i % 23) - 11) * 0.25f);
    }
    half *dSrc = nullptr;
    half *dOut = nullptr;
    if (!CheckCuda(cudaMalloc(&dSrc, src.size() * sizeof(half)), "cudaMalloc dSrc swizzle rt")) return false;
    if (!CheckCuda(cudaMalloc(&dOut, src.size() * sizeof(half)), "cudaMalloc dOut swizzle rt")) return false;
    if (!CheckCuda(cudaMemcpy(dSrc, src.data(), src.size() * sizeof(half), cudaMemcpyHostToDevice), "copy src swizzle rt")) return false;
    KernelTLoadStoreGpuSwizzle<half, Rows, Cols><<<1, 1>>>(dOut, dSrc);
    if (!CheckCuda(cudaGetLastError(), "launch tloadstore swizzle")) return false;
    if (!CheckCuda(cudaDeviceSynchronize(), "sync tloadstore swizzle")) return false;
    std::vector<half> actual(src.size());
    if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(half), cudaMemcpyDeviceToHost), "copy out swizzle rt")) return false;
    cudaFree(dSrc);
    cudaFree(dOut);
    return ExpectVecEq(src, actual, "gpu_swizzle_roundtrip");
}

bool TestGpuSwizzleTAddRoundTripMatchesReference()
{
    constexpr int Rows = 16, Cols = 16;
    std::vector<half> a(Rows * Cols), b(Rows * Cols);
    std::vector<half> expected(Rows * Cols);
    for (int i = 0; i < Rows * Cols; ++i) {
        a[i] = __float2half(static_cast<float>((i % 13) - 6) * 0.5f);
        b[i] = __float2half(static_cast<float>((i % 7) - 3) * 0.25f);
        expected[i] = __hadd(a[i], b[i]);
    }
    half *dA = nullptr;
    half *dB = nullptr;
    half *dOut = nullptr;
    if (!CheckCuda(cudaMalloc(&dA, a.size() * sizeof(half)), "cudaMalloc dA swizzle add")) return false;
    if (!CheckCuda(cudaMalloc(&dB, b.size() * sizeof(half)), "cudaMalloc dB swizzle add")) return false;
    if (!CheckCuda(cudaMalloc(&dOut, expected.size() * sizeof(half)), "cudaMalloc dOut swizzle add")) return false;
    if (!CheckCuda(cudaMemcpy(dA, a.data(), a.size() * sizeof(half), cudaMemcpyHostToDevice), "copy a swizzle add")) return false;
    if (!CheckCuda(cudaMemcpy(dB, b.data(), b.size() * sizeof(half), cudaMemcpyHostToDevice), "copy b swizzle add")) return false;
    KernelTAddGpuSwizzleRoundTrip<half, Rows, Cols><<<1, 1>>>(dOut, dA, dB);
    if (!CheckCuda(cudaGetLastError(), "launch tadd swizzle")) return false;
    if (!CheckCuda(cudaDeviceSynchronize(), "sync tadd swizzle")) return false;
    std::vector<half> actual(expected.size());
    if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(half), cudaMemcpyDeviceToHost), "copy out swizzle add")) return false;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dOut);
    return ExpectVecEq(expected, actual, "gpu_swizzle_tadd");
}

bool TestSm121FloatInlinePtxMatmulMatchesReference()
{
    constexpr int M = 8, K = 8, N = 8;
    std::vector<float> a(M * K);
    std::vector<float> b(K * N);
    for (int i = 0; i < M * K; ++i) {
        a[i] = static_cast<float>((i % 9) - 4);
    }
    for (int i = 0; i < K * N; ++i) {
        b[i] = static_cast<float>((i % 7) - 3);
    }
    auto expected = RefMatmulF32(a, b, M, K, N);
    float *dA = nullptr;
    float *dB = nullptr;
    float *dOut = nullptr;
    if (!CheckCuda(cudaMalloc(&dA, a.size() * sizeof(float)), "cudaMalloc dA")) return false;
    if (!CheckCuda(cudaMalloc(&dB, b.size() * sizeof(float)), "cudaMalloc dB")) return false;
    if (!CheckCuda(cudaMalloc(&dOut, expected.size() * sizeof(float)), "cudaMalloc dOut")) return false;
    if (!CheckCuda(cudaMemcpy(dA, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice), "copy a")) return false;
    if (!CheckCuda(cudaMemcpy(dB, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice), "copy b")) return false;
    if (!CheckCuda(cudaMemset(dOut, 0, expected.size() * sizeof(float)), "memset out")) return false;
    KernelTMATMUL<float, M, K, N><<<1, 1>>>(dOut, dA, dB);
    if (!CheckCuda(cudaGetLastError(), "launch tmatmul float")) return false;
    if (!CheckCuda(cudaDeviceSynchronize(), "sync tmatmul float")) return false;
    std::vector<float> actual(expected.size());
    if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out")) return false;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dOut);
    return ExpectVecEq(expected, actual, "tmatmul_sm121_float");
}

bool TestSm121HalfTensorCoreMatmulExtendedMatchesReference()
{
    constexpr int M = 32, K = 16, N = 32;
    std::vector<half> a(M * K);
    std::vector<half> b(K * N);
    for (int i = 0; i < M * K; ++i) {
        a[i] = __float2half(static_cast<float>((i % 13) - 6) * 0.25f);
    }
    for (int i = 0; i < K * N; ++i) {
        b[i] = __float2half(static_cast<float>((i % 11) - 5) * 0.5f);
    }
    auto expected = RefMatmulToFloat(a, b, M, K, N);
    half *dA = nullptr;
    half *dB = nullptr;
    float *dOut = nullptr;
    if (!CheckCuda(cudaMalloc(&dA, a.size() * sizeof(half)), "cudaMalloc dA half")) return false;
    if (!CheckCuda(cudaMalloc(&dB, b.size() * sizeof(half)), "cudaMalloc dB half")) return false;
    if (!CheckCuda(cudaMalloc(&dOut, expected.size() * sizeof(float)), "cudaMalloc dOut half")) return false;
    if (!CheckCuda(cudaMemcpy(dA, a.data(), a.size() * sizeof(half), cudaMemcpyHostToDevice), "copy a half")) return false;
    if (!CheckCuda(cudaMemcpy(dB, b.data(), b.size() * sizeof(half), cudaMemcpyHostToDevice), "copy b half")) return false;
    if (!CheckCuda(cudaMemset(dOut, 0, expected.size() * sizeof(float)), "memset out half")) return false;
    KernelTMATMUL<half, M, K, N><<<1, 32>>>(dOut, dA, dB);
    if (!CheckCuda(cudaGetLastError(), "launch tmatmul half")) return false;
    if (!CheckCuda(cudaDeviceSynchronize(), "sync tmatmul half")) return false;
    std::vector<float> actual(expected.size());
    if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out half")) return false;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dOut);
    return ExpectVecNear(expected, actual, "tmatmul_sm121_half_extended", 1e-3f);
}

bool TestSm121Bfloat16TensorCoreMatmulExtendedMatchesReference()
{
    constexpr int M = 16, K = 32, N = 16;
    std::vector<bfloat16_t> a(M * K);
    std::vector<bfloat16_t> b(K * N);
    for (int i = 0; i < M * K; ++i) {
        a[i] = __float2bfloat16(static_cast<float>((i % 9) - 4) * 0.75f);
    }
    for (int i = 0; i < K * N; ++i) {
        b[i] = __float2bfloat16(static_cast<float>((i % 7) - 3) * 0.5f);
    }
    auto expected = RefMatmulToFloat(a, b, M, K, N);
    bfloat16_t *dA = nullptr;
    bfloat16_t *dB = nullptr;
    float *dOut = nullptr;
    if (!CheckCuda(cudaMalloc(&dA, a.size() * sizeof(bfloat16_t)), "cudaMalloc dA bf16")) return false;
    if (!CheckCuda(cudaMalloc(&dB, b.size() * sizeof(bfloat16_t)), "cudaMalloc dB bf16")) return false;
    if (!CheckCuda(cudaMalloc(&dOut, expected.size() * sizeof(float)), "cudaMalloc dOut bf16")) return false;
    if (!CheckCuda(cudaMemcpy(dA, a.data(), a.size() * sizeof(bfloat16_t), cudaMemcpyHostToDevice), "copy a bf16")) return false;
    if (!CheckCuda(cudaMemcpy(dB, b.data(), b.size() * sizeof(bfloat16_t), cudaMemcpyHostToDevice), "copy b bf16")) return false;
    if (!CheckCuda(cudaMemset(dOut, 0, expected.size() * sizeof(float)), "memset out bf16")) return false;
    KernelTMATMUL<bfloat16_t, M, K, N><<<1, 32>>>(dOut, dA, dB);
    if (!CheckCuda(cudaGetLastError(), "launch tmatmul bf16")) return false;
    if (!CheckCuda(cudaDeviceSynchronize(), "sync tmatmul bf16")) return false;
    std::vector<float> actual(expected.size());
    if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out bf16")) return false;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dOut);
    return ExpectVecNear(expected, actual, "tmatmul_sm121_bf16_extended", 2e-2f);
}

bool TestSm121HalfTensorCoreMatmulLarge64MatchesReference()
{
    constexpr int M = 64, K = 64, N = 64;
    std::vector<half> a(M * K);
    std::vector<half> b(K * N);
    for (int i = 0; i < M * K; ++i) {
        a[i] = __float2half(static_cast<float>((i % 19) - 9) * 0.125f);
    }
    for (int i = 0; i < K * N; ++i) {
        b[i] = __float2half(static_cast<float>((i % 17) - 8) * 0.1875f);
    }
    auto expected = RefMatmulToFloat(a, b, M, K, N);
    half *dA = nullptr;
    half *dB = nullptr;
    float *dOut = nullptr;
    if (!CheckCuda(cudaMalloc(&dA, a.size() * sizeof(half)), "cudaMalloc dA half large")) return false;
    if (!CheckCuda(cudaMalloc(&dB, b.size() * sizeof(half)), "cudaMalloc dB half large")) return false;
    if (!CheckCuda(cudaMalloc(&dOut, expected.size() * sizeof(float)), "cudaMalloc dOut half large")) return false;
    if (!CheckCuda(cudaMemcpy(dA, a.data(), a.size() * sizeof(half), cudaMemcpyHostToDevice), "copy a half large")) return false;
    if (!CheckCuda(cudaMemcpy(dB, b.data(), b.size() * sizeof(half), cudaMemcpyHostToDevice), "copy b half large")) return false;
    if (!CheckCuda(cudaMemset(dOut, 0, expected.size() * sizeof(float)), "memset out half large")) return false;
    KernelTMATMUL<half, M, K, N><<<1, 32>>>(dOut, dA, dB);
    if (!CheckCuda(cudaGetLastError(), "launch tmatmul half large")) return false;
    if (!CheckCuda(cudaDeviceSynchronize(), "sync tmatmul half large")) return false;
    std::vector<float> actual(expected.size());
    if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out half large")) return false;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dOut);
    return ExpectVecNear(expected, actual, "tmatmul_sm121_half_large64", 2e-3f);
}

bool TestSm121Bfloat16TensorCoreMatmulLarge64MatchesReference()
{
    constexpr int M = 64, K = 64, N = 64;
    std::vector<bfloat16_t> a(M * K);
    std::vector<bfloat16_t> b(K * N);
    for (int i = 0; i < M * K; ++i) {
        a[i] = __float2bfloat16(static_cast<float>((i % 15) - 7) * 0.25f);
    }
    for (int i = 0; i < K * N; ++i) {
        b[i] = __float2bfloat16(static_cast<float>((i % 13) - 6) * 0.3125f);
    }
    auto expected = RefMatmulToFloat(a, b, M, K, N);
    bfloat16_t *dA = nullptr;
    bfloat16_t *dB = nullptr;
    float *dOut = nullptr;
    if (!CheckCuda(cudaMalloc(&dA, a.size() * sizeof(bfloat16_t)), "cudaMalloc dA bf16 large")) return false;
    if (!CheckCuda(cudaMalloc(&dB, b.size() * sizeof(bfloat16_t)), "cudaMalloc dB bf16 large")) return false;
    if (!CheckCuda(cudaMalloc(&dOut, expected.size() * sizeof(float)), "cudaMalloc dOut bf16 large")) return false;
    if (!CheckCuda(cudaMemcpy(dA, a.data(), a.size() * sizeof(bfloat16_t), cudaMemcpyHostToDevice), "copy a bf16 large")) return false;
    if (!CheckCuda(cudaMemcpy(dB, b.data(), b.size() * sizeof(bfloat16_t), cudaMemcpyHostToDevice), "copy b bf16 large")) return false;
    if (!CheckCuda(cudaMemset(dOut, 0, expected.size() * sizeof(float)), "memset out bf16 large")) return false;
    KernelTMATMUL<bfloat16_t, M, K, N><<<1, 32>>>(dOut, dA, dB);
    if (!CheckCuda(cudaGetLastError(), "launch tmatmul bf16 large")) return false;
    if (!CheckCuda(cudaDeviceSynchronize(), "sync tmatmul bf16 large")) return false;
    std::vector<float> actual(expected.size());
    if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out bf16 large")) return false;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dOut);
    return ExpectVecNear(expected, actual, "tmatmul_sm121_bf16_large64", 3e-2f);
}

bool TestSm121HalfTensorCoreMatmulAccMatchesReference()
{
    constexpr int M = 16, K = 16, N = 16;
    std::vector<half> a(M * K);
    std::vector<half> b(K * N);
    std::vector<float> acc(M * N);
    for (int i = 0; i < M * K; ++i) {
        a[i] = __float2half(static_cast<float>((i % 15) - 7) * 0.125f);
    }
    for (int i = 0; i < K * N; ++i) {
        b[i] = __float2half(static_cast<float>((i % 10) - 4) * 0.375f);
    }
    for (int i = 0; i < M * N; ++i) {
        acc[i] = static_cast<float>((i % 8) - 4) * 0.5f;
    }
    auto expected = RefMatmulAccToFloat(a, b, acc, M, K, N);
    half *dA = nullptr;
    half *dB = nullptr;
    float *dAcc = nullptr;
    float *dOut = nullptr;
    if (!CheckCuda(cudaMalloc(&dA, a.size() * sizeof(half)), "cudaMalloc dA acc")) return false;
    if (!CheckCuda(cudaMalloc(&dB, b.size() * sizeof(half)), "cudaMalloc dB acc")) return false;
    if (!CheckCuda(cudaMalloc(&dAcc, acc.size() * sizeof(float)), "cudaMalloc dAcc acc")) return false;
    if (!CheckCuda(cudaMalloc(&dOut, expected.size() * sizeof(float)), "cudaMalloc dOut acc")) return false;
    if (!CheckCuda(cudaMemcpy(dA, a.data(), a.size() * sizeof(half), cudaMemcpyHostToDevice), "copy a acc")) return false;
    if (!CheckCuda(cudaMemcpy(dB, b.data(), b.size() * sizeof(half), cudaMemcpyHostToDevice), "copy b acc")) return false;
    if (!CheckCuda(cudaMemcpy(dAcc, acc.data(), acc.size() * sizeof(float), cudaMemcpyHostToDevice), "copy acc in")) return false;
    KernelTMATMUL_ACC<half, M, K, N><<<1, 32>>>(dOut, dAcc, dA, dB);
    if (!CheckCuda(cudaGetLastError(), "launch tmatmul acc")) return false;
    if (!CheckCuda(cudaDeviceSynchronize(), "sync tmatmul acc")) return false;
    std::vector<float> actual(expected.size());
    if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out acc")) return false;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dAcc);
    cudaFree(dOut);
    return ExpectVecNear(expected, actual, "tmatmul_sm121_acc", 1e-3f);
}

bool TestSm121Bfloat16TensorCoreMatmulBiasMatchesReference()
{
    constexpr int M = 16, K = 16, N = 16;
    std::vector<bfloat16_t> a(M * K);
    std::vector<bfloat16_t> b(K * N);
    std::vector<float> bias(N);
    for (int i = 0; i < M * K; ++i) {
        a[i] = __float2bfloat16(static_cast<float>((i % 9) - 4) * 0.25f);
    }
    for (int i = 0; i < K * N; ++i) {
        b[i] = __float2bfloat16(static_cast<float>((i % 7) - 3) * 0.75f);
    }
    for (int i = 0; i < N; ++i) {
        bias[i] = static_cast<float>(i - 8) * 0.125f;
    }
    auto expected = RefMatmulBiasToFloat(a, b, bias, M, K, N);
    bfloat16_t *dA = nullptr;
    bfloat16_t *dB = nullptr;
    float *dBias = nullptr;
    float *dOut = nullptr;
    if (!CheckCuda(cudaMalloc(&dA, a.size() * sizeof(bfloat16_t)), "cudaMalloc dA bias")) return false;
    if (!CheckCuda(cudaMalloc(&dB, b.size() * sizeof(bfloat16_t)), "cudaMalloc dB bias")) return false;
    if (!CheckCuda(cudaMalloc(&dBias, bias.size() * sizeof(float)), "cudaMalloc dBias")) return false;
    if (!CheckCuda(cudaMalloc(&dOut, expected.size() * sizeof(float)), "cudaMalloc dOut bias")) return false;
    if (!CheckCuda(cudaMemcpy(dA, a.data(), a.size() * sizeof(bfloat16_t), cudaMemcpyHostToDevice), "copy a bias")) return false;
    if (!CheckCuda(cudaMemcpy(dB, b.data(), b.size() * sizeof(bfloat16_t), cudaMemcpyHostToDevice), "copy b bias")) return false;
    if (!CheckCuda(cudaMemcpy(dBias, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice), "copy bias")) return false;
    if (!CheckCuda(cudaMemset(dOut, 0, expected.size() * sizeof(float)), "memset out bias")) return false;
    KernelTMATMUL_BIAS<bfloat16_t, M, K, N><<<1, 32>>>(dOut, dA, dB, dBias);
    if (!CheckCuda(cudaGetLastError(), "launch tmatmul bias")) return false;
    if (!CheckCuda(cudaDeviceSynchronize(), "sync tmatmul bias")) return false;
    std::vector<float> actual(expected.size());
    if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out bias")) return false;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dBias);
    cudaFree(dOut);
    return ExpectVecNear(expected, actual, "tmatmul_sm121_bias", 2e-2f);
}

bool TestSm121HalfTMATMUL_MXMatchesReference()
{
    constexpr int M = 16, K = 16, N = 16;
    std::vector<half> a(M * K);
    std::vector<half> b(K * N);
    std::vector<float> aScale(K, 1.0f), bScale(N, 1.0f);
    for (int i = 0; i < M * K; ++i) a[i] = __float2half(static_cast<float>((i % 13) - 6) * 0.25f);
    for (int i = 0; i < K * N; ++i) b[i] = __float2half(static_cast<float>((i % 11) - 5) * 0.5f);
    auto expected = RefMatmulToFloat(a, b, M, K, N);
    half *dA = nullptr; half *dB = nullptr; float *dAS = nullptr; float *dBS = nullptr; float *dOut = nullptr;
    if (!CheckCuda(cudaMalloc(&dA, a.size() * sizeof(half)), "cudaMalloc dA mx")) return false;
    if (!CheckCuda(cudaMalloc(&dB, b.size() * sizeof(half)), "cudaMalloc dB mx")) return false;
    if (!CheckCuda(cudaMalloc(&dAS, aScale.size() * sizeof(float)), "cudaMalloc dAS mx")) return false;
    if (!CheckCuda(cudaMalloc(&dBS, bScale.size() * sizeof(float)), "cudaMalloc dBS mx")) return false;
    if (!CheckCuda(cudaMalloc(&dOut, expected.size() * sizeof(float)), "cudaMalloc dOut mx")) return false;
    if (!CheckCuda(cudaMemcpy(dA, a.data(), a.size() * sizeof(half), cudaMemcpyHostToDevice), "copy a mx")) return false;
    if (!CheckCuda(cudaMemcpy(dB, b.data(), b.size() * sizeof(half), cudaMemcpyHostToDevice), "copy b mx")) return false;
    if (!CheckCuda(cudaMemcpy(dAS, aScale.data(), aScale.size() * sizeof(float), cudaMemcpyHostToDevice), "copy as mx")) return false;
    if (!CheckCuda(cudaMemcpy(dBS, bScale.data(), bScale.size() * sizeof(float), cudaMemcpyHostToDevice), "copy bs mx")) return false;
    if (!CheckCuda(cudaMemset(dOut, 0, expected.size() * sizeof(float)), "memset out mx")) return false;
    KernelTMATMUL_MX<half, M, K, N><<<1, 32>>>(dOut, dA, dAS, dB, dBS);
    if (!CheckCuda(cudaGetLastError(), "launch tmatmul_mx")) return false;
    if (!CheckCuda(cudaDeviceSynchronize(), "sync tmatmul_mx")) return false;
    std::vector<float> actual(expected.size());
    if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out mx")) return false;
    cudaFree(dA); cudaFree(dB); cudaFree(dAS); cudaFree(dBS); cudaFree(dOut);
    return ExpectVecNear(expected, actual, "tmatmul_mx", 1e-3f);
}

bool TestSm121HalfTGEMV_MXMatchesReference()
{
    constexpr int M = 16, K = 16, N = 16;
    std::vector<half> a(M * K);
    std::vector<half> b(K * N);
    std::vector<float> aScale(K, 1.0f), bScale(N, 1.0f);
    for (int i = 0; i < M * K; ++i) a[i] = __float2half(static_cast<float>((i % 9) - 4) * 0.375f);
    for (int i = 0; i < K * N; ++i) b[i] = __float2half(static_cast<float>((i % 7) - 3) * 0.625f);
    auto expected = RefMatmulToFloat(a, b, M, K, N);
    half *dA = nullptr; half *dB = nullptr; float *dAS = nullptr; float *dBS = nullptr; float *dOut = nullptr;
    if (!CheckCuda(cudaMalloc(&dA, a.size() * sizeof(half)), "cudaMalloc dA gemv_mx")) return false;
    if (!CheckCuda(cudaMalloc(&dB, b.size() * sizeof(half)), "cudaMalloc dB gemv_mx")) return false;
    if (!CheckCuda(cudaMalloc(&dAS, aScale.size() * sizeof(float)), "cudaMalloc dAS gemv_mx")) return false;
    if (!CheckCuda(cudaMalloc(&dBS, bScale.size() * sizeof(float)), "cudaMalloc dBS gemv_mx")) return false;
    if (!CheckCuda(cudaMalloc(&dOut, expected.size() * sizeof(float)), "cudaMalloc dOut gemv_mx")) return false;
    if (!CheckCuda(cudaMemcpy(dA, a.data(), a.size() * sizeof(half), cudaMemcpyHostToDevice), "copy a gemv_mx")) return false;
    if (!CheckCuda(cudaMemcpy(dB, b.data(), b.size() * sizeof(half), cudaMemcpyHostToDevice), "copy b gemv_mx")) return false;
    if (!CheckCuda(cudaMemcpy(dAS, aScale.data(), aScale.size() * sizeof(float), cudaMemcpyHostToDevice), "copy as gemv_mx")) return false;
    if (!CheckCuda(cudaMemcpy(dBS, bScale.data(), bScale.size() * sizeof(float), cudaMemcpyHostToDevice), "copy bs gemv_mx")) return false;
    if (!CheckCuda(cudaMemset(dOut, 0, expected.size() * sizeof(float)), "memset out gemv_mx")) return false;
    KernelTGEMV_MX<half, M, K, N><<<1, 32>>>(dOut, dA, dAS, dB, dBS);
    if (!CheckCuda(cudaGetLastError(), "launch tgemv_mx")) return false;
    if (!CheckCuda(cudaDeviceSynchronize(), "sync tgemv_mx")) return false;
    std::vector<float> actual(expected.size());
    if (!CheckCuda(cudaMemcpy(actual.data(), dOut, actual.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out gemv_mx")) return false;
    cudaFree(dA); cudaFree(dB); cudaFree(dAS); cudaFree(dBS); cudaFree(dOut);
    return ExpectVecNear(expected, actual, "tgemv_mx", 1e-3f);
}

} // namespace

int main()
{
    int failed = 0;
    auto run = [&](const char *name, bool (*fn)()) {
        const bool ok = fn();
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << std::endl;
        if (!ok) {
            ++failed;
        }
    };

    run("TLoadNdRowMajorMatchesReference", &TestTLoadNdRowMajorMatchesReference);
    run("TLoadDnColMajorMatchesReference", &TestTLoadDnColMajorMatchesReference);
    run("TStoreNdAndDnMatchReference", &TestTStoreNdAndDnMatchReference);
    run("TAddMatchesReference", &TestTAddMatchesReference);
    run("GpuSwizzlePhysicalLayoutMatchesReference", &TestGpuSwizzlePhysicalLayoutMatchesReference);
    run("GpuSwizzleRoundTripMatchesReference", &TestGpuSwizzleRoundTripMatchesReference);
    run("GpuSwizzleTAddRoundTripMatchesReference", &TestGpuSwizzleTAddRoundTripMatchesReference);
    run("Sm121FloatInlinePtxMatmulMatchesReference", &TestSm121FloatInlinePtxMatmulMatchesReference);
    run("Sm121HalfTensorCoreMatmulExtendedMatchesReference", &TestSm121HalfTensorCoreMatmulExtendedMatchesReference);
    run("Sm121Bfloat16TensorCoreMatmulExtendedMatchesReference", &TestSm121Bfloat16TensorCoreMatmulExtendedMatchesReference);
    run("Sm121HalfTensorCoreMatmulLarge64MatchesReference", &TestSm121HalfTensorCoreMatmulLarge64MatchesReference);
    run("Sm121Bfloat16TensorCoreMatmulLarge64MatchesReference", &TestSm121Bfloat16TensorCoreMatmulLarge64MatchesReference);
    run("Sm121HalfTensorCoreMatmulAccMatchesReference", &TestSm121HalfTensorCoreMatmulAccMatchesReference);
    run("Sm121Bfloat16TensorCoreMatmulBiasMatchesReference", &TestSm121Bfloat16TensorCoreMatmulBiasMatchesReference);
    run("Sm121HalfTMATMUL_MXMatchesReference", &TestSm121HalfTMATMUL_MXMatchesReference);
    run("Sm121HalfTGEMV_MXMatchesReference", &TestSm121HalfTGEMV_MXMatchesReference);

    return failed == 0 ? 0 : 1;
}
