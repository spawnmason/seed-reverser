#include <iostream>
#include <fstream>
#include <vector>
#include <charconv>
#include <chrono>
#include <thread>
#include <bitset>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <endian.h>

void printCudaVersion() {
    int cudaVersion;
    cudaRuntimeGetVersion(&cudaVersion);
    printf("CUDA runtime version: %d.%d\n", cudaVersion / 1000, (cudaVersion % 100) / 10);
    auto err = cudaDriverGetVersion(&cudaVersion);
    printf("CUDA driver version: %d.%d\n", cudaVersion / 1000, (cudaVersion % 100) / 10);
}

void checkCudaError(cudaError_t err) {
    if (err) {
        std::cerr << "CUDA error " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

std::vector<int64_t> readSeeds(char* path) {
    auto file = std::ifstream{path};
    std::vector<int64_t> seeds;
    std::string line;
    while (std::getline(file, line)) {
        int64_t num;
        if (std::from_chars(line.c_str(), line.c_str() + line.size(), num).ec != std::errc{}) {
            std::cerr << "Failed to parse \"" << line << "\" as a uint64" << std::endl;
            exit(1);
        }
        seeds.push_back(num);
    }
    return seeds;
}

struct Int3 {
    int i:24;
} __attribute__((packed));

constexpr int64_t OVERWORLD_2B2T_SEED = -4172144997902289642LL;
constexpr int64_t WOODLAND_BOUNDS = 23440;
constexpr int64_t MANSIONS_IN_ROW = (WOODLAND_BOUNDS * 2 + 1);
constexpr int64_t TOTAL_MANSIONS = MANSIONS_IN_ROW * MANSIONS_IN_ROW;
constexpr int64_t END_CITY_BOUNDS = WOODLAND_BOUNDS * 4;
constexpr int64_t CITIES_IN_ROW = (END_CITY_BOUNDS * 2 + 1);
constexpr int64_t TOTAL_CITIES = CITIES_IN_ROW * CITIES_IN_ROW;

constexpr int64_t chunkZMultInv248 = 211541297333629LL; // https://www.wolframalpha.com/input?i=132897987541%5E-1+mod+2%5E48
constexpr auto RAND_MASK = (1LL << 48) - 1;
constexpr auto LOWER_32_MASK = (1LL << 32) - 1;
constexpr auto LOWER_36_MASK = (1LL << 36) - 1;
constexpr auto TABLE_SIZE = (1LL << 32);
constexpr auto SIZEOF_TABLE = TABLE_SIZE * sizeof(int16_t);
constexpr auto END_TABLE_SIZE = (1LL << 36);
constexpr auto SIZEOF_END_TABLE = END_TABLE_SIZE * sizeof(Int3);

__device__ __host__ int64_t generalizedWoodlandMansionSeed(int chunkX, int chunkZ) { // from World and WoodlandMansion
    return ((int64_t) chunkX * 341873128712LL + (int64_t) chunkZ * 132897987541LL + (int64_t) 10387319) & RAND_MASK; // rand drops highest 16 bits anyway
}
__device__ __host__ int64_t generalizedEndCitySeed(int chunkX, int chunkZ) { // from World and end city
    return ((int64_t) chunkX * 341873128712LL + (int64_t) chunkZ * 132897987541LL + (int64_t) 10387313) & RAND_MASK; // rand drops highest 16 bits anyway
}

__global__ void writeToSeedTable(int16_t* table) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_MANSIONS) return;
    auto x = tid / MANSIONS_IN_ROW - WOODLAND_BOUNDS;
    auto z = tid % MANSIONS_IN_ROW - WOODLAND_BOUNDS;
    auto seed = generalizedWoodlandMansionSeed(x, z);
    table[seed & LOWER_32_MASK] = (int16_t) x;
}

__global__ void writeToEndSeedTable(Int3* table, int64_t min, int64_t maxExclusive) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    if (tid >= TOTAL_CITIES) return;
    auto x = tid / CITIES_IN_ROW - END_CITY_BOUNDS;
    auto z = tid % CITIES_IN_ROW - END_CITY_BOUNDS;
    auto seed = generalizedEndCitySeed(x, z);
    auto index = seed & LOWER_36_MASK;
    if (index >= min && index < maxExclusive) {
        table[index - min] = Int3{(int) x};
    }
}

int16_t* createSeedTableGpu() {
    auto t1 = std::chrono::steady_clock::now();
    int16_t *deviceTablePtr;
    auto err = cudaMalloc(&deviceTablePtr, SIZEOF_TABLE);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate table on gpu (" << cudaGetErrorString(err) << ")" << std::endl;
        exit(1);
    }
    cudaMemset(deviceTablePtr, 0, SIZEOF_TABLE);
    const auto threadsPerBlock = 256;
    const auto numBlocks = (TOTAL_MANSIONS + threadsPerBlock - 1) / threadsPerBlock;
    writeToSeedTable<<<numBlocks, threadsPerBlock>>>(deviceTablePtr);
    cudaDeviceSynchronize();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1);
    std::cerr << "generated mansion table in " << millis.count() << "ms" << std::endl;
    return deviceTablePtr;
}

struct Result {
    int64_t startSeed;
    int32_t x;
    int32_t z;
    int32_t steps;
};

__device__ __host__ int64_t prevSeed(int64_t seed) {
    return (seed - 0xBLL) * 0xdfe05bcb1365LL & RAND_MASK;
}

__device__ __host__ int64_t reverseWoodlandZGivenX(int64_t seed48Bits, int chunkX) {
    return (((seed48Bits - 10387319 - (int64_t) chunkX * 341873128712LL)) * chunkZMultInv248) << 16 >> 16;
}

int64_t reverseEndZGivenX(int64_t seed48Bits, int chunkX) {
    return (((seed48Bits - 10387313 - (int64_t) chunkX * 341873128712LL)) * chunkZMultInv248) << 16 >> 16;
}

__device__ __host__ bool isWoodlandSeed(const int16_t* table, int64_t seed, int& xOut, int64_t& zOut, int64_t worldSeed) {
    const int64_t generalizedSeed = (seed - worldSeed) & RAND_MASK;
    const auto x = table[generalizedSeed & LOWER_32_MASK];
    const auto z = reverseWoodlandZGivenX(generalizedSeed, x);
    xOut = x;
    zOut = z;
    return z >= -WOODLAND_BOUNDS & z <= WOODLAND_BOUNDS;
}

bool endValidSlow(int64_t seed, int& xOut, int64_t& zOut, int64_t worldSeed) {
    const int64_t generalizedSeed = (seed - worldSeed) & RAND_MASK;
    for (int x = -END_CITY_BOUNDS; x <= END_CITY_BOUNDS; x++) {
        int64_t z = reverseEndZGivenX(generalizedSeed, x);
        if (z >= -END_CITY_BOUNDS && z <= END_CITY_BOUNDS) {
            xOut = x;
            zOut = z;
            return true;
        }
    }
    return false;
}

bool isEndSeed(const Int3* table, int64_t seed, int& xOut, int64_t& zOut, int64_t worldSeed, int steps) {
    const int64_t generalizedSeed = (seed - worldSeed) & RAND_MASK;
    const auto x = table[generalizedSeed & LOWER_36_MASK].i;
    const auto z = reverseEndZGivenX(generalizedSeed, x);
    xOut = x;
    zOut = z;
    if (steps == 6313) {
        //std::cout << "seed=" << seed << std::endl;
        std::cout << "x=" << x << " z=" << z << std::endl;
        exit(0);
    }
    return z >= -END_CITY_BOUNDS & z <= END_CITY_BOUNDS;
}

__global__ void computeSteps(const int16_t* mansionTable, int64_t worldSeed, const int64_t* seedsArr, Result* resultArr, size_t numData) {
    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    [[unlikely]] if (tid >= numData) {
        return;
    }
    //printf("tid = %d, offset=%d, size=%d\n", tid, offset, size);

    const auto startSeed = seedsArr[tid];
    auto seed = startSeed;
    int steps = 0;
    int x;
    int64_t z;
    while (!isWoodlandSeed(mansionTable, seed ^ 0x5DEECE66DLL, x, z, worldSeed)) {
        seed = prevSeed(seed);
        steps++;
    }
    //printf("Found mansion at %d,%ld, steps = %d\n", x, z, steps);
    resultArr[tid] = {.startSeed = startSeed, .x = (int16_t) x, .z = (int16_t) z, .steps = steps};
}

template<typename T>
T* uploadBuffer(T* data, size_t len) {
    void* deviceBuffer;
    checkCudaError(cudaMalloc(&deviceBuffer, len * sizeof(T)));
    checkCudaError(cudaMemcpy(deviceBuffer, data, len * sizeof(T), cudaMemcpyHostToDevice));
    return (T*) deviceBuffer;
}


bool fileExists(char* path) {
    return access(path, F_OK) == 0;
}

bool readBytes(int socket, void* bufferIn, size_t size) {
    auto buffer = (char*) bufferIn;
    ssize_t bytesRead = 0;
    ssize_t result;
    while (bytesRead < size) {
        result = read(socket, buffer + bytesRead, size - bytesRead);
        if (result < 1 ) {
            return false;
        }
        bytesRead += result;
    }
    return true;
}
bool writeBytes(int socket, const void* bufferIn, size_t size) {
    auto buffer = (const char*) bufferIn;
    ssize_t bytesRead = 0;
    ssize_t result;
    while (bytesRead < size) {
        result = write(socket, buffer + bytesRead, size - bytesRead);
        if (result < 1) {
            return false;
        }
        bytesRead += result;
    }
    return true;
}

bool writeI16be(int fd, int16_t x) {
    auto be = htobe16(x);
    return writeBytes(fd, &be, 2);
}
bool writeI32be(int fd, int32_t x) {
    auto be = htobe32(x);
    return writeBytes(fd, &be, 4);
}
bool writeI64be(int fd, int64_t x) {
    auto be = htobe64(x);
    return writeBytes(fd, &be, 8);
}

std::vector<Result> reverseOverworldSeeds(const int16_t* deviceMansionTable, const std::vector<int64_t>& seeds, int64_t worldSeed) {
    auto startTime = std::chrono::steady_clock::now();

    const auto* deviceInputSeeds = uploadBuffer(seeds.data(), seeds.size());
    Result* deviceResults;
    checkCudaError(cudaMalloc(&deviceResults, seeds.size() * sizeof(Result)));

    const auto threadsPerBlock = 256;
    const auto numBlocks = (seeds.size() + threadsPerBlock - 1) / threadsPerBlock;
    computeSteps<<<numBlocks, threadsPerBlock>>>(deviceMansionTable, worldSeed, deviceInputSeeds, deviceResults, seeds.size());
    cudaDeviceSynchronize();

    std::vector<Result> results(seeds.size());
    checkCudaError(cudaMemcpy(results.data(), deviceResults, seeds.size() * sizeof(Result), cudaMemcpyDeviceToHost));
    cudaFree(deviceResults);
    auto execTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
    std::cerr << "Reversed " << seeds.size() << " seeds in " << execTime << "ms" << std::endl;

    return results;
}

Result reverseEndSeed(const Int3* table, int64_t seed, int64_t worldSeed) {
    const auto startSeed = seed;
    int steps = 0;
    int x;
    int64_t z;
    //while (!isEndSeed(table, seed ^ 0x5DEECE66DLL, x, z, worldSeed, steps)) {
    while (!endValidSlow(seed ^ 0x5DEECE66DLL, x, z, worldSeed)) {
        seed = prevSeed(seed);
        steps++;
    }
    std::cout << "x=" << x << " z=" << z << std::endl;
    return {.startSeed = startSeed, .x = x, .z = (int) z, .steps = steps};
}

std::vector<Result> reverseEndSeeds(const Int3* endTable, const std::vector<int64_t>& seeds, int64_t worldSeed) {
    auto startTime = std::chrono::steady_clock::now();
    std::vector<Result> results;
    results.reserve(seeds.size());
    std::cout << "reversing end seeds..." << std::endl;
    for (auto seed : seeds) {
        auto result = reverseEndSeed(endTable, seed, worldSeed);
        results.push_back(result);
        std::cout << "end result: " << result.steps << std::endl;
    }
    auto execTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
    std::cerr << "Reversed " << seeds.size() << " end seeds in " << execTime << "ms" << std::endl;
    return results;
}

void* mmapFile(int fd, int64_t size, int prot) {
    struct stat sb{};
    if (fstat(fd, &sb) == -1) {
        std::cerr << "Failed to get file size\n";
        exit(1);
    }
    const auto statSize = (size_t) sb.st_size;
    if (statSize != size) {
        std::cerr << "table exists but is not " << size << " bytes" << std::endl;
        exit(1);
    }

    void* addr = mmap(nullptr, size, prot, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        std::cerr << "Failed to mmap file\n";
        exit(1);
    }
    return addr;
}

void writeOverworldTable(int16_t* table) {
    for (int16_t x = -WOODLAND_BOUNDS; x <= WOODLAND_BOUNDS; x++) {
        for (int16_t z = -WOODLAND_BOUNDS; z <= WOODLAND_BOUNDS; z++) {
            auto seed = generalizedWoodlandMansionSeed(x, z);
            table[seed & LOWER_32_MASK] = x;
        }
    }
}
void writeEndTable(Int3* table, int64_t min, int64_t maxExclusive, int i) {
    auto startTime = std::chrono::steady_clock::now();

    for (int x = -END_CITY_BOUNDS; x <= END_CITY_BOUNDS; x++) {
        if (x % 1000 == 0) std::cout << "x = " << x << ", i = " << i << std::endl;
        for (int z = -END_CITY_BOUNDS; z <= END_CITY_BOUNDS; z++) {
            auto seed = generalizedEndCitySeed(x, z);
            auto index = seed & LOWER_36_MASK;
            if (index >= min && index < maxExclusive) {
                table[index - min] = Int3{x};
            }
        }
    }
    auto execTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
    std::cerr << "Wrote end table in " << execTime << "ms" << std::endl;
}

void bigWrite(int fd, const char* bytes, size_t size) {
    size_t written = 0;
    while (written < size) {
        auto w = write(fd, bytes + written, size - written);
        if (w == -1) {
            std::cerr << "write failed " << strerror(errno) << std::endl;
            exit(1);
        }
        written += w;
    }
}

void writeEndTableGpuAsync(Int3* deviceMem, int64_t chunkSizeBytes, int64_t min, int64_t maxExclusive) {
    cudaMemset(deviceMem, 0, chunkSizeBytes);
    const auto threadsPerBlock = 256;
    const auto numBlocks = (TOTAL_CITIES + threadsPerBlock - 1) / threadsPerBlock;

    writeToEndSeedTable<<<numBlocks, threadsPerBlock>>>(deviceMem, min, maxExclusive);
}

const Int3* getOrCreateEndCityTable() {
    int fd = open("end_table", O_RDONLY);
    if (fd == -1) {
        if (errno == ENOENT) {
            std::cerr << "end_table doesn't exist, creating it now..." << std::endl;
            fd = open("end_table_temp", O_CREAT | O_TRUNC | O_WRONLY, 0666);
            if (fd == -1) {
                std::cerr << "fail xd" << std::endl;
                exit(1);
            }
            if (ftruncate(fd, SIZEOF_END_TABLE) == -1) {
                std::cerr << "Failed to set ftruncate() to " << SIZEOF_END_TABLE << std::endl;
                exit(1);
            }
            constexpr int64_t chunkSize = (1LL << 36) / 32;
            auto* buffer = new Int3[chunkSize];
            Int3* deviceBuffer;
            auto err = cudaMalloc(&deviceBuffer, chunkSize * sizeof(Int3));
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate buffer on gpu (" << cudaGetErrorString(err) << ")" << std::endl;
                exit(1);
            }
            // start the computation so the first iteration has a result (this is sort of like the -1st iteration)
            writeEndTableGpuAsync(deviceBuffer, chunkSize * sizeof(Int3), 0, chunkSize);
            for (int i = 0; i < 32; i++) {
                std::cout << "creating end city table " << (i + 1) << "/32" << std::endl;
                // synchronized the call from the previous iteration
                cudaDeviceSynchronize();
                cudaMemcpy(buffer, deviceBuffer, chunkSize * sizeof(Int3), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                if (i < 32 - 1) {
                    auto min = chunkSize * (i + 1);
                    auto max = min + chunkSize;
                    // start the computation for the next iteration, this will run at the same time as the write to disk
                    writeEndTableGpuAsync(deviceBuffer, chunkSize * sizeof(Int3), min, max);
                    bigWrite(fd, (const char*) buffer, chunkSize * sizeof(Int3));
                } else {
                    // on the last iteration we don't need the gpu to do anything
                    bigWrite(fd, (const char*) buffer, chunkSize * sizeof(Int3));
                }
            }
            cudaFree(deviceBuffer);
            delete[] buffer;
            close(fd);
            if (rename("end_table_temp", "end_table") == -1) {
                std::cerr << "failed to rename end_table_temp to end_table" << strerror(errno) << std::endl;
                exit(1);
            }
            // we created the file, the happy path should work now
            return getOrCreateEndCityTable();
        } else {
            std::cerr << "Failed to open end_table (" << strerror(errno) << ")" << std::endl;
            exit(1);
        }
    }
    return (const Int3*) mmapFile(fd, SIZEOF_END_TABLE, PROT_READ);
}

int32_t reverseSeedsFromAPI(const int16_t* deviceMansionTable, const Int3* endTable) {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(3460);
    inet_pton(AF_INET, "192.168.69.1", &server_addr.sin_addr);
    if (connect(sockfd, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Failed to connect to nitwit" << std::endl;
        close(sockfd);
        return 0;
    }
    const auto token = "def7fa9dd0c96bc1dcd2913d19120b7c89c026407fd67053774d316bd90b3459815479eef495f7f93f8f845c544515b77360b60e1214ed786820cb4d94e6abcd v3";
    writeI16be(sockfd, (int16_t) strlen(token));
    writeBytes(sockfd, token, strlen(token));
    int8_t numWorlds;
    if (!readBytes(sockfd, &numWorlds, 1)) {
        std::cerr << "failed to read number of worlds";
        close(sockfd);
        return 0;
    }
    int totalResults = 0;
    for (int i = 0; i < numWorlds; i++ ) {
        int8_t dimension;
        if (!readBytes(sockfd, &dimension, 1)) {
            std::cerr << "failed to read dimension";
            close(sockfd);
            return 0;
        }
        if (dimension != 0 && dimension != 1) {
            close(sockfd);
            return 0;
        }
        int64_t worldSeed;
        if (!readBytes(sockfd, &worldSeed, 8)) {
            std::cerr << "failed to read world seed";
            close(sockfd);
            return 0;
        }
        worldSeed = be64toh(worldSeed);
        int32_t seedCount;
        if (!readBytes(sockfd, &seedCount, 4)) {
            std::cerr << "failed to read seed count" << std::endl;
            close(sockfd);
            return 0;
        }
        seedCount = ntohl(seedCount);
        std::cerr << "nitwit gave us " << seedCount << " seeds for dimension " << (int)dimension << std::endl;
        if (seedCount == 0) continue; // important to match behavior or master
        std::vector<int64_t> seeds(seedCount);
        auto seedBytes = seeds.size() * sizeof(int64_t);
        if (!readBytes(sockfd, seeds.data(), seedBytes)) {
            std::cerr << "Failed to read seed bytes" << std::endl;
            close(sockfd);
            return 0;
        }
        for (auto& x : seeds) {
            x = be64toh(x);
        }
        std::vector<Result> results;
        if (dimension == 0) {
           results = reverseOverworldSeeds(deviceMansionTable, seeds, worldSeed);
        } else {
            results = reverseEndSeeds(endTable, seeds, worldSeed);
        }
        if (!writeI32be(sockfd, (int32_t) results.size())) {
            std::cerr << "Failed to results" << std::endl;
            close(sockfd);
            return 0;
        }
        for (const auto& r : results) {
            if (!writeI64be(sockfd, r.startSeed) ||
                !writeI32be(sockfd, r.steps) ||
                !writeI16be(sockfd, r.x) ||
                !writeI16be(sockfd, r.z))
            {
                std::cerr << "Failed to write seed" << std::endl;
                close(sockfd);
                return 0;
            }
        }
        totalResults += results.size();
    }

    close(sockfd);

    std::cerr << "sent " << totalResults << " results to nitwit" << std::endl;
    return totalResults;
}


int createEmptyFile(const char* fileName, int64_t fileSize) {
    FILE* fp;
    fp = fopen(fileName, "wb");
    if(fp == NULL){
        std::cout << "Failed to create file" << std::endl;
        return -1; //error opening file
    }
    fseek(fp, fileSize-1, SEEK_SET);
    fputc('\0', fp);
    fclose(fp);
    return 0; //success
}

void dumpTableToFile(const int16_t* table) {
    std::ofstream out("mansion_table", std::ios::out | std::ios::binary | std::ios::trunc);
    out.write((const char*)table, SIZEOF_TABLE);
}

Result reverseSeedCpu(const int16_t* table, int64_t seed) {
    const auto startSeed = seed;
    int steps = 0;
    int x;
    int64_t z;
    while (!isWoodlandSeed(table, seed ^ 0x5DEECE66DLL, x, z, OVERWORLD_2B2T_SEED)) {
        seed = prevSeed(seed);
        steps++;
    }
    //printf("Found mansion at %d,%ld, steps = %d\n", x, z, steps);
    return {.startSeed = startSeed, .x = (int16_t) x, .z = (int16_t) z, .steps = steps};
}

std::vector<Result> reverseSeedsCpu(const int16_t* table, const std::vector<int64_t>& seeds) {
    auto startTime = std::chrono::steady_clock::now();
    std::vector<Result> results{};
    for (const auto s : seeds) {
        results.push_back(reverseSeedCpu(table, s));
    }

    auto execTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime).count();
    std::cerr << "Reversed " << seeds.size() << " seeds in " << execTime << "ms" << std::endl;

    return results;
}

int main(int argc, char** argv) {
    printCudaVersion();
    const Int3* endCityTable = getOrCreateEndCityTable();
    // x=78568 z=5572
    // end result: 6313
    /*auto seed = generalizedEndCitySeed(78568, 5572);
    int xOut;
    int64_t zOut;
    if (endValidSlow(seed, xOut, zOut, 1434823964849314312LL)) {
        std::cout << "is end seed (x=" << xOut << " z=" << zOut << ")" << std::endl;
    }
    auto index = seed & LOWER_36_MASK;
    auto x = endCityTable[index].i;
    std::cout << x << std::endl;
    return 0;*/
    const int16_t* deviceMansionTable = createSeedTableGpu();
    if (argc <= 1) {
        while (true) {
            const auto count = reverseSeedsFromAPI(deviceMansionTable, endCityTable);
            if (count < 1000) {
                std::this_thread::sleep_for(std::chrono::seconds{120});
            }
        }
    } else {
        if (argc != 3) {
            std::cerr << "Expected 2 file arguments" << std::endl;
            return 1;
        }
        auto seeds = readSeeds(argv[1]);
        if (!fileExists(argv[2])) {
            std::cerr << "Output file doesnt exist" << std::endl;
            return 1;
        }
        auto results = reverseOverworldSeeds(deviceMansionTable, seeds, OVERWORLD_2B2T_SEED);
        std::ofstream out(argv[2]);
        for (const auto& r : results) {
            out << r.startSeed << ", " << r.steps << ", " << r.x << ", " << r.z << std::endl;
        }
    }
}
