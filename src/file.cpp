#include <fstream>
#include <vector>
#include "../include/file.h"

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

std::vector<uint8_t> readFileInt(const std::string& filename) {
    std::basic_ifstream<uint8_t> file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<uint8_t> buffer(fileSize);

    file.seekg(0);
    buffer.assign(std::istreambuf_iterator<uint8_t>(file), std::istreambuf_iterator<uint8_t>());
    file.close();

    return buffer;
}

bool writeFile(
    const char* fileName,
    const uint8_t* data,
    int size
) {
    FILE* f = std::fopen(fileName, "wb");

    if (f == nullptr) {
        return false;
    }

    std::string result;

    size_t write_result = std::fwrite(data, sizeof(uint8_t), size, f);

    if (write_result < static_cast<size_t>(size)) {
        return false;
    }

    fclose(f);

    return true;
}
