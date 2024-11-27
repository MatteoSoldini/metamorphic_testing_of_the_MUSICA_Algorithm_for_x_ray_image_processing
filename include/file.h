#pragma once
#include <vector>
#include <string>

std::vector<char> readFile(const std::string& filename);

std::vector<uint8_t> readFileInt(const std::string& filename);

bool writeFile(
    const char* fileName,
    const uint8_t* data,
    int size
);