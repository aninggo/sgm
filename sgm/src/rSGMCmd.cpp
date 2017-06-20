// Copyright © Robert Spangenberg, 2014.
// See license.txt for more details
// This is an example for Git
#include "iostream"

#include <smmintrin.h> // intrinsics
#include <emmintrin.h>

#include "StereoBMHelper.h"
#include "FastFilters.h"
#include "StereoSGM.h"

#include <vector>
#include <list>
#include <algorithm>
#include <numeric>

#include "MyImage.h"

void correctEndianness(uint16* input, uint16* output, uint32 size)
{
    uint8* outputByte = (uint8*)output;
    uint8* inputByte = (uint8*)input;

    for (uint32 i=0; i < size; i++) {
        *(outputByte+1) = *inputByte;
        *(outputByte) = *(inputByte+1);
        outputByte+=2;
        inputByte+=2;
    }
}

template <typename T>
void census5x5_t_SSE(T* source, uint32* dest, uint32 width, uint32 height)
{

}

template <>
void census5x5_t_SSE(uint16* source, uint32* dest, uint32 width, uint32 height)
{
    census5x5_16bit_SSE(source, dest, width, height);
}



template<typename T>
void processCensus5x5SGM(T* leftImg, T* rightImg, float32* output, float32* dispImgRight,
                         int width, int height, uint16 paths, const int numThreads, const int numStrips, const int dispCount)
{
    const int maxDisp = dispCount - 1;

    std::cout << std::endl << "- "  << ", " << paths << ", " << numThreads << ", " << numStrips << ", " << dispCount << std::endl;

    // get memory and init sgm params
    uint32* leftImgCensus = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);
    uint32* rightImgCensus = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);

    StereoSGMParams_t params;
    params.lrCheck = true;
    params.MedianFilter = true;
    params.Paths = paths;
    params.subPixelRefine = 0;
    params.NoPasses = 2;
    params.rlCheck = false;

    const int maxDisp2 = 79;
    const sint32 dispSubsample = 4;
    uint16* dsi = (uint16*)_mm_malloc(width*height*(maxDisp2 + 1)*sizeof(uint16), 32);

    StripedStereoSGM<T> stripedStereoSGM(width, height, maxDisp2, numStrips, 16, params);

    census5x5_t_SSE(leftImg, leftImgCensus, width, height);
    census5x5_t_SSE(rightImg, rightImgCensus, width, height);

    costMeasureCensusCompressed5x5_xyd_SSE(leftImgCensus, rightImgCensus, height, width, dispCount, params.InvalidDispCost, dispSubsample, dsi, numThreads);

    stripedStereoSGM.process(leftImg, output, dispImgRight, dsi, numThreads);

    uncompressDisparities_SSE(output, width, height, dispSubsample);

    _mm_free(dsi);


}

int main(int argc, char **argv)
{
    const int verbose = 1;

    // load parameter
    if (argc!=4) {
        std::cout << "expected imL,imR,disp as params"<< std::endl;
        return -1;
    }
    char *im1name = argv[1];
    char *im2name = argv[2];
    char *dispname= argv[3];

    fillPopCount16LUT();

    // load images
    MyImage<uint16> myImg1, myImg2;
    readPGM(myImg1, im1name);
    readPGM(myImg2, im2name);

    std::cout << "image 1 " << myImg1.getWidth() << "x" << myImg1.getHeight() << std::endl;
    std::cout << "image 2 " << myImg2.getWidth() << "x" << myImg2.getHeight() << std::endl;

    if (myImg1.getWidth() % 16 != 0) {
        std::cout << "Image width must be a multiple of 16" << std::endl;
        return 0;
    }

    MyImage<uint8> disp(myImg1.getWidth(), myImg1.getHeight());

    uint16* leftImg = (uint16*)_mm_malloc(myImg1.getWidth()*myImg1.getHeight()*sizeof(uint16), 16);
    uint16* rightImg = (uint16*)_mm_malloc(myImg2.getWidth()*myImg2.getHeight()*sizeof(uint16), 16);
    correctEndianness((uint16*)myImg1.getData(), leftImg, myImg1.getWidth()*myImg1.getHeight());
    correctEndianness((uint16*)myImg2.getData(), rightImg, myImg1.getWidth()*myImg1.getHeight());

    float32* dispImg = (float32*)_mm_malloc(myImg1.getWidth()*myImg1.getHeight()*sizeof(float32), 16);
    float32* dispImgRight = (float32*)_mm_malloc(myImg1.getWidth()*myImg1.getHeight()*sizeof(float32), 16);

    // start processing

    // striped SGM, 4 threads, disparity compression with sub-sampling 4 (only implemented for 128)
    processCensus5x5SGM(leftImg, rightImg, dispImg, dispImgRight, myImg1.getWidth(), myImg1.getHeight(), 8, 4, 4, 128);

    // write output
    uint8* dispOut = disp.getData();
    for (uint32 i = 0; i < myImg1.getWidth()*myImg1.getHeight(); i++) {
        if (dispImg[i]>0) {
            dispOut[i] = (uint8)dispImg[i];
        }
        else {
            dispOut[i] = 0;
        }
    }

    std::string dispnamePlusDisp = dispname;
    writePGM(disp, dispnamePlusDisp.c_str(), verbose);

    // cleanup
    _mm_free(leftImg);
    _mm_free(rightImg);
    _mm_free(dispImg);
    _mm_free(dispImgRight);

    return 0;
}
