#include <stdio.h>
#include <process.h>
#include <string>
#include <fstream>
#include "../../include/vk_processing.h"

#define ASSERT_MSG(cond, msg)                               \
    if (!(cond)) {                                      \
        fprintf(stderr, "MAIN ERROR: %s\n", msg); \
        exit(1);                                        \
    }

bool readFileInt(const std::string& filename, std::vector<uint8_t>& buffer) {
    std::basic_ifstream<uint8_t> file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        return false;
    }

    size_t fileSize = (size_t)file.tellg();
    buffer.resize(fileSize);

    file.seekg(0);
    buffer.assign(std::istreambuf_iterator<uint8_t>(file), std::istreambuf_iterator<uint8_t>());
    file.close();

    return true;
}

int main( int argc, char* argv[] ) {
    static const uint32_t imageSize = 3072;
    
    for (uint32_t i = 0; i < argc; i++) {
        printf("%d = %s\n", i, argv[i]);
    }

    ASSERT_MSG(argc == 3, "wrong number of arguments");

    std::string rawFile = argv[1];
    std::string outFile = argv[2];

    printf("raw file %s\n", rawFile.c_str());
    printf("out file %s\n", outFile.c_str());

    VulkanState* vkState = new VulkanState();
    ASSERT_MSG(vkState->init(std::vector<const char*>()), "failed to initialize vulkan");

    VulkanProcessing* vkProcessing = new VulkanProcessing(vkState);

    std::vector<VkImageView> outImageViews = {};
    ASSERT_MSG(vkProcessing->init(imageSize, &outImageViews), "failed to initialize vk processing");

    // load raw file
    std::vector<uint8_t> imageData;
    ASSERT_MSG(readFileInt(rawFile, imageData), "failed to load file");

    const uint32_t offset = 256;
    const uint32_t expectedDataSize = offset + (imageSize * imageSize) * 2;

    ASSERT_MSG(imageData.size() == expectedDataSize, "the image data don't match the actual image size");

    std::vector<uint16_t> pixels;
    pixels.resize(imageSize * imageSize);

    uint32_t index = offset;

    for (int x = 0; x < imageSize; x++) {
        for (int y = 0; y < imageSize; y++) {
            //ASSERT_MSG(index > imageData.size(), "error reading data")

            pixels[x * imageSize + y] =
                (imageData[index + 1] << 8) | (imageData[index] & 0xff);
            index += 2;
        }
    }

    ASSERT_MSG(vkProcessing->execute(pixels.data()), "processing failed");

    ASSERT_MSG(vkProcessing->saveOutImage(outFile), "failed to save out image");

#ifndef NDEBUG
    ASSERT_MSG(vkProcessing->debugProcess(), "failed to debug process");
#endif

    ASSERT_MSG(vkProcessing->cleanup(), "failed to cleanup");
    vkState->cleanup();
}