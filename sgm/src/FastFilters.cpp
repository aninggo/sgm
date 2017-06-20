// Copyright © Robert Spangenberg, 2014.
// See license.txt for more details

#include "StereoCommon.h"
#include "FastFilters.h"

#include <smmintrin.h> // intrinsics
#include <emmintrin.h>
#include <iostream>
#include "string.h" // memset
#include "assert.h"

inline uint8* getPixel8(uint8* base, uint32 width, int j, int i)
{
    return base+i*width+j;
}

inline uint16* getPixel16(uint16* base, uint32 width, int j, int i)
{
    return base+i*width+j;
}

inline uint32* getPixel32(uint32* base, uint32 width, int j, int i)
{
    return base+i*width+j;
}

inline uint64* getPixel64(uint64* base, uint32 width, int j, int i)
{
    return base+i*width+j;
}

FORCEINLINE void testpixel(uint8* source, uint32 width, sint32 i, sint32 j,uint64& value, sint32 x, sint32 y)
{
    if (*getPixel8(source, width,j+x,i+y) - *getPixel8(source, width,j-x,i-y)>0) {
        value += 1;
    }
}

FORCEINLINE void testpixel2(uint8* source, uint32 width, sint32 i, sint32 j,uint64& value, sint32 x, sint32 y)
{
    value *= 2;
    uint8 result = *getPixel8(source, width,j+x,i+y) - *getPixel8(source, width,j-x,i-y)>0;
    value += result;
}

FORCEINLINE void testpixel_16bit(uint16* source, uint32 width, sint32 i, sint32 j, uint64& value, sint32 x, sint32 y)
{
    if (*getPixel16(source, width, j + x, i + y) - *getPixel16(source, width, j - x, i - y) > 0) {
        value += 1;
    }
}

FORCEINLINE void testpixel2_16bit(uint16* source, uint32 width, sint32 i, sint32 j, uint64& value, sint32 x, sint32 y)
{
    value *= 2;
    uint8 result = *getPixel16(source, width, j + x, i + y) - *getPixel16(source, width, j - x, i - y) > 0;
    value += result;
}


// census 5x5
// input uint16 image, output uint32 image
// 2.56ms/2, 768x480, 9.2 cycles/pixel
void census5x5_16bit_SSE(uint16* source, uint32* dest, uint32 width, uint32 height)
{
    uint32* dst = dest;
    uint16* src = source;
    
    // memsets just for the upper and lower two lines, not really necessary
    memset(dest, 0, width*2*sizeof(uint32));
    memset(dest+width*(height-2), 0, width*2*sizeof(uint32));

    // input lines 0,1,2
    uint16* i0 = src;
    uint16* i1 = src+width;
    uint16* i2 = src+2*width;
    uint16* i3 = src+3*width;
    uint16* i4 = src+4*width;

    // output at first result
    uint32* result = dst + 2*width+2;
    const uint16* const end_input = src + width*height;


    for(; i4+4<=end_input; i0 +=1, i1 +=1 ,i2 +=1, i3 +=1 ,i4 +=1){
        uint16 pixelcv =*(i2 +2);

        //line1
        uint16 pixel0h = (*i0  >=pixelcv) ? 0:65535u;

        uint16 pixel1v =*(i0 +1);
        uint16 pixel1h =(pixel1v >=pixelcv ) ? 0:65535u;

        uint16 pixel2v =*(i0 +2);
        uint16 pixel2h =(pixel2v >=pixelcv ) ? 0:65535u;

        uint16 pixel3v =*(i0 +3);
        uint16 pixel3h =(pixel3v >=pixelcv ) ? 0:65535u;

        uint16 pixel4v =*(i0 +4);
        uint16 pixel4h =(pixel4v >=pixelcv ) ? 0:65535u;

        //line2
        uint16 pixel5h =(*i1 >=pixelcv ) ? 0:65535u;

        uint16 pixel6v =*(i1 +1);
        uint16 pixel6h =(pixel6v >=pixelcv ) ? 0:65535u;

        uint16 pixel7v =*(i1 +2);
        uint16 pixel7h =(pixel7v >=pixelcv ) ? 0:65535u;

        uint16 pixel8v =*(i1 +3);
        uint16 pixel8h =(pixel8v >=pixelcv ) ? 0:65535u;

        uint16 pixel9v =*(i1 +4);
        uint16 pixel9h =(pixel9v >=pixelcv ) ? 0:65535u;

        //line3
        uint16 pixel10h =(*i2  >=pixelcv ) ? 0:65535u;

        uint16 pixel11v =*(i2 +1);
        uint16 pixel11h =(pixel11v >=pixelcv ) ? 0:65535u;

        uint16 pixel12v =*(i2 +3);
        uint16 pixel12h =(pixel12v >=pixelcv ) ? 0:65535u;

        uint16 pixel13v =*(i2 +4);
        uint16 pixel13h =(pixel13v >=pixelcv ) ? 0:65535u;

        //line4
        uint16 pixel14h =(*i3 >=pixelcv) ? 0:65535u;

        uint16 pixel15v =*(i3 +1);
        uint16 pixel15h =(pixel15v >=pixelcv ) ? 0:65535u;

        uint16 B1B2mask0 = 32768u;
        uint16 B1B2mask1 = 16384u;
        uint16 B1B2mask2 = 8192u;
        uint16 B1B2mask3 = 4096u;
        uint16 B1B2mask4 = 2048u;
        uint16 B1B2mask5 = 1024u;
        uint16 B1B2mask6 = 512u;
        uint16 B1B2mask7 = 256u;
        uint16 B1mask8 = 128u;
        uint16 B1mask9 = 64u;
        uint16 B1mask10 = 32u;
        uint16 B1mask11 = 16u;
        uint16 B1mask12 = 8u;
        uint16 B1mask13 = 4u;
        uint16 B1mask14 = 2u;
        uint16 B1mask15 = 1u;

        uint16 resultByte1 = pixel0h & B1mask8;
        resultByte1 = resultByte1|(pixel1h&B1mask9);
        resultByte1 = resultByte1|(pixel2h&B1mask10);
        resultByte1 = resultByte1|(pixel3h&B1mask11);
        resultByte1 = resultByte1|(pixel4h&B1mask12);
        resultByte1 = resultByte1|(pixel5h&B1mask13);
        resultByte1 = resultByte1|(pixel6h&B1mask14);
        resultByte1 = resultByte1|(pixel7h&B1mask15);
        uint16 setByte1zero = 255u;
        resultByte1 = resultByte1& setByte1zero;

        uint16 pixel16v =*(i3 +2);
        uint16 pixel16h =(pixel16v >=pixelcv ) ? 0:65535u;

        uint16 pixel17v =*(i3 +3);
        uint16 pixel17h =(pixel17v >=pixelcv ) ? 0:65535u;

        uint16 pixel18v =*(i3 +4);
        uint16 pixel18h =(pixel18v >=pixelcv ) ? 0:65535u;

        //line5
        uint16 pixel19h =(*i4  >=pixelcv ) ? 0:65535u;

        uint16 pixel20v =*(i4 +1);
        uint16 pixel20h =(pixel20v >=pixelcv ) ? 0:65535u;

        uint16 pixel21v =*(i4 +2);
        uint16 pixel21h =(pixel21v >=pixelcv ) ? 0:65535u;

        uint16 pixel22v =*(i4 +3);
        uint16 pixel22h =(pixel22v >=pixelcv ) ? 0:65535u;

        uint16 pixel23v =*(i4 +4);
        uint16 pixel23h =(pixel23v >=pixelcv ) ? 0:65535u;

        uint16 resultByte2 = pixel8h & B1B2mask0;
        resultByte2 = resultByte2|(pixel9h&B1B2mask1);
        resultByte2 = resultByte2|(pixel10h&B1B2mask2);
        resultByte2 = resultByte2|(pixel11h&B1B2mask3);
        resultByte2 = resultByte2|(pixel12h&B1B2mask4);
        resultByte2 = resultByte2|(pixel13h&B1B2mask5);
        resultByte2 = resultByte2|(pixel14h&B1B2mask6);
        resultByte2 = resultByte2|(pixel15h&B1B2mask7);
        resultByte2 = resultByte2|(pixel16h&B1mask8);
        resultByte2 = resultByte2|(pixel17h&B1mask9);
        resultByte2 = resultByte2|(pixel18h&B1mask10);
        resultByte2 = resultByte2|(pixel19h&B1mask11);
        resultByte2 = resultByte2|(pixel20h&B1mask12);
        resultByte2 = resultByte2|(pixel21h&B1mask13);
        resultByte2 = resultByte2|(pixel22h&B1mask14);
        resultByte2 = resultByte2|(pixel23h&B1mask15);

        uint16* result16 = (uint16*)result;
        *result16 = resultByte1;
        *(result16+1) =resultByte2;
        result =(result+1);
    }
}

inline void vecSortandSwap(__m128& a, __m128& b)
{
    __m128 temp = a;
    a = _mm_min_ps(a,b);
    b = _mm_max_ps(temp,b);
}

void median3x3_SSE(float32* source, float32* dest, uint32 width, uint32 height)
{
    // check width restriction
    assert(width % 4 == 0);
    
    float32* destStart = dest;
    //  lines
    float32* line1 = source;
    float32* line2 = source + width;
    float32* line3 = source + 2*width;

    float32* end = source + width*height;

    dest += width;
    __m128 lastMedian = _mm_setzero_ps();

    do {
        // fill value
        const __m128 l1_reg = _mm_load_ps(line1);
        const __m128 l1_reg_next = _mm_load_ps(line1+4);
        __m128 v0 = l1_reg;
        __m128 v1 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l1_reg_next),_mm_castps_si128(l1_reg), 4));
        __m128 v2 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l1_reg_next),_mm_castps_si128(l1_reg), 8));

        const __m128 l2_reg = _mm_load_ps(line2);
        const __m128 l2_reg_next = _mm_load_ps(line2+4);
        __m128 v3 = l2_reg;
        __m128 v4 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l2_reg_next),_mm_castps_si128(l2_reg), 4));
        __m128 v5 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l2_reg_next),_mm_castps_si128(l2_reg), 8));

        const __m128 l3_reg = _mm_load_ps(line3);
        const __m128 l3_reg_next = _mm_load_ps(line3+4);
        __m128 v6 = l3_reg;
        __m128 v7 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l3_reg_next),_mm_castps_si128(l3_reg), 4));
        __m128 v8 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(l3_reg_next),_mm_castps_si128(l3_reg), 8));

        // find median through sorting network
        vecSortandSwap(v1, v2) ; vecSortandSwap(v4, v5) ; vecSortandSwap(v7, v8) ;
        vecSortandSwap(v0, v1) ; vecSortandSwap(v3, v4) ; vecSortandSwap(v6, v7) ;
        vecSortandSwap(v1, v2) ; vecSortandSwap(v4, v5) ; vecSortandSwap(v7, v8) ;
        vecSortandSwap(v0, v3) ; vecSortandSwap(v5, v8) ; vecSortandSwap(v4, v7) ;
        vecSortandSwap(v3, v6) ; vecSortandSwap(v1, v4) ; vecSortandSwap(v2, v5) ;
        vecSortandSwap(v4, v7) ; vecSortandSwap(v4, v2) ; vecSortandSwap(v6, v4) ;
        vecSortandSwap(v4, v2) ;

        // comply to alignment restrictions
        const __m128i c = _mm_alignr_epi8(_mm_castps_si128(v4), _mm_castps_si128(lastMedian), 12);
        _mm_store_si128((__m128i*)dest, c);
        lastMedian = v4;

        dest+=4; line1+=4; line2+=4; line3+=4;

    } while (line3+4+4 <= end);

    memcpy(destStart, source, sizeof(float32)*(width+1));
    memcpy(destStart+width*height-width-1-3, source+width*height-width-1-3, sizeof(float32)*(width+1+3));
}

