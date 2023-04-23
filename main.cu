#include <iostream>
#include <fstream>
#include <vector>
#include <charconv>
#include <chrono>
#include <cassert>
#include <optional>
#include <fstream>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

void printCudaVersion() {
    int cudaVersion;
    cudaRuntimeGetVersion(&cudaVersion);
    printf("CUDA runtime version: %d\n", cudaVersion);
    auto err = cudaDriverGetVersion(&cudaVersion);
    printf("CUDA driver version: %d.%d\n", cudaVersion / 1000, (cudaVersion % 100) / 10);
}

std::vector<int64_t> readSeeds(char* path) {
    auto file = std::ifstream{path};
    std::vector<int64_t> seeds;
    std::string line;
    while (std::getline(file, line)) {
        int64_t num;
        if (std::from_chars(line.c_str(), line.c_str() + line.size(), num).ec != std::errc{}) {
            std::cout << "Failed to parse \"" << line << "\" as a uint64" << std::endl;
            exit(1);
        }
        seeds.push_back(num);
    }
    return seeds;
}

constexpr int64_t WORLD_SEED = -4172144997902289642LL;
constexpr int64_t WOODLAND_BOUNDS = 23440;
//constexpr int64_t TOTAL_MANSIONS = (WOODLAND_BOUNDS * 2 + 1) * (WOODLAND_BOUNDS * 2 + 1);
constexpr int64_t chunkZMultInv248 = 211541297333629LL; // https://www.wolframalpha.com/input?i=132897987541%5E-1+mod+2%5E48
constexpr auto RAND_MASK = (1LL << 48) - 1;
constexpr auto LOWER_32_MASK = (1LL << 32) - 1;
constexpr auto INDEX_SIZE = (1LL << 32);

int64_t woodlandMansionSeed(int chunkX, int chunkZ) { // from World and WoodlandMansion
    return ((int64_t) chunkX * 341873128712LL + (int64_t) chunkZ * 132897987541LL + WORLD_SEED + (int64_t) 10387319) & RAND_MASK; // rand drops highest 16 bits anyway
}

int64_t nextSeed(int64_t seed) {
    return (seed * 0x5DEECE66DLL + 0xBLL) & RAND_MASK;
}

int64_t prevSeedHost(int64_t seed) {
    return (seed - 0xBLL) * 0xdfe05bcb1365LL & RAND_MASK;
}

int64_t reverseWoodlandZGivenXHost(int64_t seed48Bits, int chunkX) {
    constexpr auto mask = (1LL << 48) - 1;
    return (((seed48Bits - WORLD_SEED - 10387319 - (int64_t) chunkX * 341873128712LL) & mask) * chunkZMultInv248) << 16 >> 16;
}

std::optional<std::pair<int, int>> woodlandValid(int64_t candidate) {
    for (int x = -WOODLAND_BOUNDS; x <= WOODLAND_BOUNDS; x++) {
        long z = reverseWoodlandZGivenXHost(candidate, x);
        if (z >= -WOODLAND_BOUNDS && z <= WOODLAND_BOUNDS) {
            return {{x, z}};
        }
    }
    return {};
}

std::vector<int16_t> createSeedTable() {
    std::cout << "Creating seed table" << std::endl;
    auto t1 = std::chrono::steady_clock::now();
    // Initialize with max 16 bit int, which is out of the range of valid mansion coords
    std::vector<int16_t> out(INDEX_SIZE, 0);

    for (int x = -WOODLAND_BOUNDS; x <= WOODLAND_BOUNDS; x++) {
        for (int z = -WOODLAND_BOUNDS; z <= WOODLAND_BOUNDS; z++) {
            auto seed = woodlandMansionSeed(x, z);
            out[seed & LOWER_32_MASK] = (int16_t) x;
        }
    }
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1);
    std::cout << "Created seed table in " << millis.count() << "ms" << std::endl;

    return out;
}

struct LcgParams {
    int64_t mult, add;
    int64_t steps;
};

LcgParams computeLcgJump(const int64_t stepsIn) {
    int64_t steps = stepsIn & RAND_MASK;
    LcgParams result{1, 0, steps};
    LcgParams current{0x5DEECE66DL, 0xB};

    while (steps > 0) {
        if (steps % 2 == 1) {
            result.mult = (result.mult * current.mult);
            result.add = (current.add + result.add * current.mult);
        }
        steps /= 2;

        int64_t mult = current.mult;
        int64_t add = current.add;
        current = {(mult * mult), (add * mult + add)};
    }

    result.mult &= RAND_MASK;
    result.add &= RAND_MASK;
    return {result.mult, result.add, stepsIn};
}

struct Result {
    int x;
    int z;
    int64_t steps;
    int64_t startSeed;
};

__device__ int64_t prevSeed(int64_t seed) {
    return (seed - 0xBLL) * 0xdfe05bcb1365LL & RAND_MASK;
}

__device__ int64_t reverseWoodlandZGivenX(int64_t seed48Bits, int chunkX) {
    return (((seed48Bits - WORLD_SEED - 10387319 - (int64_t) chunkX * 341873128712LL)) * chunkZMultInv248) << 16 >> 16;
}

__device__ void getRange(int tid, int numThreads, int dataSize, int& offset, int& size) {
    const int totalSize = dataSize;
    const int numFullBlocks = totalSize / numThreads;
    const int numOverflow = totalSize % numThreads;

    if (tid < numOverflow) {
        offset = tid * (numFullBlocks + 1);
        size = numFullBlocks + 1;
    } else {
        offset = tid * numFullBlocks + numOverflow;
        size = numFullBlocks;
    }
}

__device__ bool isWoodlandSeed(const int16_t* table, int64_t seed, int& xOut, int64_t& zOut) {
    const auto x = table[seed & LOWER_32_MASK];
    const auto z = reverseWoodlandZGivenX(seed, x);
    xOut = x;
    zOut = z;
    return z >= -WOODLAND_BOUNDS & z <= WOODLAND_BOUNDS;
}

__global__ void computeSteps(const int16_t* mansionTable, const int64_t* seedsArr, Result* resultArr, int totalSeedCount, int threadCount) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset;
    int size;
    getRange(tid, threadCount, totalSeedCount, offset, size);
    //printf("tid = %d, offset=%d, size=%d\n", tid, offset, size);
    Result prevResult{};

    for (int i = 0; i < size; i++) {
        const auto startSeed = seedsArr[offset + i];
        auto seed = startSeed;
        int steps = 0;
        int x;
        int64_t z;
        while (!isWoodlandSeed(mansionTable, seed, x, z)) {
            [[unlikely]] if (seed == prevResult.startSeed) {
                resultArr[offset + i] = {
                    .x = prevResult.x,
                    .z = prevResult.z,
                    .steps = steps + prevResult.steps,
                    .startSeed = startSeed
                };
                goto end;
            }
            seed = prevSeed(seed);
            steps++;
        }
        //printf("Found mansion at %d,%ld, steps = %d\n", x, z, steps);
        {
            Result result{.x = x, .z = (int) z, .steps = steps, .startSeed = startSeed};
            resultArr[offset + i] = result;
            prevResult = result;
        }
        end:;
    }
}

template<typename T>
T* uploadBuffer(T* data, size_t len) {
    void* deviceBuffer;
    cudaMalloc(&deviceBuffer, len * sizeof(T));
    cudaMemcpy(deviceBuffer, data, len * sizeof(T), cudaMemcpyHostToDevice);
    return (T*) deviceBuffer;
}

void verifyResult(Result result, int64_t inputSeed) {
    std::cout << "Verifying results" << std::endl;
    std::cout << "input seed: " << inputSeed << std::endl;
    std::cout << "steps = " << result.steps << std::endl;
    int64_t outSeed = inputSeed;
    for (int i = 0; i < result.steps; i++) {
        outSeed = prevSeedHost(outSeed);
    }
    std::cout << "output seed: " << outSeed << std::endl;
    auto isValid = woodlandValid(outSeed);
    if (isValid) {
        std::cout << "valid seed" << std::endl;
        std::cout << "x = " << isValid->first << "," << isValid->second << std::endl;
    } else {
        std::cout << "not a valid seed" << std::endl;
    }
}

std::optional<void*> mmapTable() {
    int fd = open("mansion_table", O_RDONLY);
    if (fd == -1) {
        return {};
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "Failed to get file size\n";
        return {};
    }
    const auto size = (size_t) sb.st_size;
    if (size != INDEX_SIZE * sizeof(int16_t)) {
        std::cerr << "table exists but is not " << INDEX_SIZE * sizeof(int16_t) << " bytes" << std::endl;
        return {};
    }

    void* addr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        std::cerr << "Failed to mmap file\n";
        return {};
    }
    return {addr};
}

int main(int argc, char** argv) {
    printCudaVersion();
    if (argc < 2) {
        std::cout << "Expected file argument" << std::endl;
        return 1;
    }
    auto seeds = readSeeds(argv[1]);

    void* deviceMansionTable;
    auto err = cudaMalloc(&deviceMansionTable, INDEX_SIZE * sizeof(int16_t));
    if (err != cudaSuccess) {
        std::cout << "Failed to allocate table on gpu (" << cudaGetErrorString(err) << ")" << std::endl;
        return 1;
    }


    std::vector<int16_t> seedTable;
    if (auto file = mmapTable(); file) {
        std::cout << "Loading seed table from disk" << std::endl;
        auto t1 = std::chrono::steady_clock::now();
        seedTable.assign(INDEX_SIZE, 0);
        memcpy(seedTable.data(), *file, INDEX_SIZE * sizeof(int16_t));
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1);
        std::cout << "Loaded seed table in " << millis.count() << "ms" << std::endl;
    } else {
        seedTable = createSeedTable();
        std::ofstream out("mansion_table", std::ios::out | std::ios::binary);
        if (!out) {
            std::cerr << "Failed to open file for writing" << std::endl;
        } else {
            out.write((const char*)seedTable.data(), seedTable.size() * sizeof(int16_t));
            out.close();
        }
    }
    assert(table.size() == INDEX_SIZE);
    std::cout << "Uploading to gpu" << std::endl;
    cudaMemcpy(deviceMansionTable, seedTable.data(), INDEX_SIZE * sizeof(int16_t), cudaMemcpyHostToDevice);

    int numMPs{};
    cudaDeviceGetAttribute(&numMPs, cudaDevAttrMultiProcessorCount, 0);
    printf("numMPs = %d\n", numMPs);
    int warpSizeUwu{};
    cudaDeviceGetAttribute(&warpSizeUwu, cudaDevAttrWarpSize, 0);
    printf("Warp size: %d\n", warpSizeUwu);
    int threadCount = numMPs * warpSizeUwu;

    auto startTime = std::chrono::steady_clock::now();

    const auto* deviceInputSeeds = uploadBuffer(seeds.data(), seeds.size());
    void* deviceResults;
    cudaMalloc(&deviceResults, seeds.size() * sizeof(Result));

    computeSteps<<<numMPs, warpSizeUwu>>>((int16_t*) deviceMansionTable, deviceInputSeeds, (Result*) deviceResults, seeds.size(), threadCount);
    cudaDeviceSynchronize();

    std::vector<Result> results(seeds.size());
    cudaMemcpy(results.data(), deviceResults, seeds.size() * sizeof(Result), cudaMemcpyDeviceToHost);

    auto execTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
    std::cout << "Reversed " << seeds.size() << " seeds in " << execTime << "ms" << std::endl;

    return 0;
}
