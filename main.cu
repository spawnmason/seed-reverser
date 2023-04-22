#include <iostream>
#include <fstream>
#include <vector>
#include <charconv>
#include <chrono>
#include <cassert>
#include <optional>

void printCudaVersion() {
    int cudaVersion;
    cudaRuntimeGetVersion(&cudaVersion);
    printf("CUDA runtime version: %d\n", cudaVersion);
    auto err = cudaDriverGetVersion(&cudaVersion);
    printf("CUDA driver version: %d.%d\n", cudaVersion / 1000, (cudaVersion % 100) / 10);
}

std::vector<uint64_t> readSeeds(char* path) {
    auto file = std::ifstream{path};
    std::vector<uint64_t> seeds;
    std::string line;
    while (std::getline(file, line)) {
        uint64_t num;
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
constexpr int64_t TOTAL_MANSIONS = (WOODLAND_BOUNDS * 2 + 1) * (WOODLAND_BOUNDS * 2 + 1);
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

int64_t prevSeed(int64_t seed) {
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
    // Initialize with max 16 bit int, which is out of the range of valid mansion coords
    std::vector<int16_t> out(INDEX_SIZE, 32767);

    for (int x = -WOODLAND_BOUNDS; x <= WOODLAND_BOUNDS; x++) {
        for (int z = -WOODLAND_BOUNDS; z <= WOODLAND_BOUNDS; z++) {
            auto seed = woodlandMansionSeed(x, z);
            out[seed & LOWER_32_MASK] = (int16_t) x;
        }
    }
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
    bool done;
    int64_t steps;
};

__device__ Result reverseResult;
__device__ int mutex;

__device__ int64_t nextSeed(int64_t seed, LcgParams lcg) {
    return (seed * lcg.mult + lcg.add) & RAND_MASK;
}

__device__ int64_t reverseWoodlandZGivenX(int64_t seed48Bits, int chunkX) {
    constexpr auto mask = (1LL << 48) - 1;
    return (((seed48Bits - WORLD_SEED - 10387319 - (int64_t) chunkX * 341873128712LL) & mask) * chunkZMultInv248) << 16 >> 16;
}

__device__ bool isInRange(int64_t x, int64_t z) {
    return x >= -WOODLAND_BOUNDS && x <= WOODLAND_BOUNDS && z >= -WOODLAND_BOUNDS && z <= WOODLAND_BOUNDS;
}

__global__ void tryComputeSteps(const int16_t* mansionTable, const LcgParams lcg, int64_t* seeds, int kernelsRan) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    //int step = gridDim.x * blockDim.x;
    auto seed = seeds[tid];
    const auto inputSeed = seed;
    //seed = nextSeed(seed, lcg);
    seeds[tid] = nextSeed(seed, lcg); // update the input seeds for the next iteration

    const auto x = mansionTable[seed & LOWER_32_MASK];
    const auto z = reverseWoodlandZGivenX(seed, x);
    if (isInRange(x, z)) {
        const auto steps = kernelsRan + tid;
        //printf("found result yay! seed: %ld , x/y: (%d, %ld), steps: %ld\n", seed, x, z, steps);
        //printf("lcg: mult=%lx, add=%lx, steps=%ld\n", lcg.mult, lcg.add, lcg.steps);
        while (atomicCAS(&mutex, 0, 1) != 0);
        auto currentResult = reverseResult;
        if (!currentResult.done || steps < currentResult.steps) {
            reverseResult = {true, steps};
        }
        atomicExch(&mutex, 0);
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
        outSeed = prevSeed(outSeed);
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

int main(int argc, char** argv) {
    printCudaVersion();
    if (argc < 2) {
        std::cout << "Expected file argument" << std::endl;
        //return 1;
    }
    //auto seeds = readSeeds(argv[1]);

    void* deviceMansionTable;
    auto err = cudaMalloc(&deviceMansionTable, INDEX_SIZE * sizeof(int16_t));
    if (err != cudaSuccess) {
        std::cout << "Failed to allocate table on gpu (" << cudaGetErrorString(err) << ")" << std::endl;
        return 1;
    }

    std::cout << "Creating seed table" << std::endl;
    auto t1 = std::chrono::steady_clock::now();
    auto table = createSeedTable();
    assert(table.size() == INDEX_SIZE);
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1);
    std::cout << "Created seed table in " << millis.count() << "ms" << std::endl;
    std::cout << "Uploading to gpu" << std::endl;
    cudaMemcpy(deviceMansionTable, table.data(), INDEX_SIZE * sizeof(int16_t), cudaMemcpyHostToDevice);

    int numMPs{};
    cudaDeviceGetAttribute(&numMPs, cudaDevAttrMultiProcessorCount, 0);
    printf("numMPs = %d\n", numMPs);
    int warpSizeUwu{};
    cudaDeviceGetAttribute(&warpSizeUwu, cudaDevAttrWarpSize, 0);
    printf("Warp size: %d\n", warpSizeUwu);
    int threadCount = numMPs * warpSizeUwu;

    const auto testSeed = 250517125837036LL;
    auto seed = testSeed;
    std::vector<int64_t> inputSeeds;
    for (int i = 0; i < threadCount; i++) {
        inputSeeds.push_back(seed);
        seed = prevSeed(seed);
    }
    auto* deviceInputSeeds = uploadBuffer(inputSeeds.data(), inputSeeds.size());

    Result result = {false, 2'000'000'000};
    cudaMemcpyToSymbol(reverseResult, &result, sizeof(Result));
    t1 = std::chrono::steady_clock::now();
    int i = 0;
    do {
        tryComputeSteps<<<numMPs, warpSizeUwu>>>((int16_t*) deviceMansionTable, computeLcgJump(-threadCount), deviceInputSeeds, i * threadCount);
        cudaDeviceSynchronize();

        cudaMemcpyFromSymbol(&result, reverseResult, sizeof(Result));
        i++;
    } while (!result.done);
    std::cout << result.steps << " steps yay!!!" << std::endl;
    auto micros = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t1);
    std::cout << "reversed seed in " << micros.count() << "ms" << std::endl;
    //verifyResult(result, testSeed);

    return 0;
}
