// Copyright © Robert Spangenberg, 2014.
// See license.txt for more details
#include "StereoCommon.h"
#include "StereoSGM.h"
#include <string.h>
// accumulate along paths
// variable P2 param
template<typename T>
template <int NPaths>
void StereoSGM<T>::accumulateVariableParamsSSE(uint16* &dsi, T* img, uint16* &S)
{
    /* Params */
    const sint32 paramP1 = m_params.P1;
    const uint16 paramInvalidDispCost = m_params.InvalidDispCost;
    const int paramNoPasses = m_params.NoPasses;
    const uint16 MAX_SGM_COST = UINT16_MAX;

    // change params for fixed, if necessary
    const float32 paramAlpha = m_params.Alpha;
    const sint32 paramGamma = m_params.Gamma;
    const sint32 paramP2min =  m_params.P2min;
    const int width = m_width;
    //const int width2 = width+2;
    const int maxDisp = m_maxDisp;
    const int height = m_height;
    const int disp = maxDisp+1;
    const int dispP2 = disp+2;
    // accumulated cost along path r
    // two extra elements for -1 and maxDisp+1 disparity
    // current and last line (or better, two buffers)
    uint16* L_r0      = ((uint16*) _mm_malloc(dispP2*sizeof(uint16),16))+1;
    uint16* L_r0_last = ((uint16*) _mm_malloc(dispP2*sizeof(uint16),16))+1;
    uint16* L_r1      = ((uint16*) _mm_malloc(width*dispP2*sizeof(uint16),16))+1;
    uint16* L_r1_last = ((uint16*) _mm_malloc(width*dispP2*sizeof(uint16),16))+1;
    uint16* L_r2      = ((uint16*) _mm_malloc(width*dispP2*sizeof(uint16),16))+1;
    uint16* L_r2_last = ((uint16*) _mm_malloc(width*dispP2*sizeof(uint16),16))+1;
    uint16* L_r3      = ((uint16*) _mm_malloc(width*dispP2*sizeof(uint16),16))+1;
    uint16* L_r3_last = ((uint16*) _mm_malloc(width* dispP2*sizeof(uint16),16))+1;

    /* image line pointers */
    T* img_line_last = NULL;
    T* img_line = NULL;
    /* left border */

    // min L_r cache
    uint16 minL_r0_Array[2];
    uint16* minL_r0 = &minL_r0_Array[0];
    uint16* minL_r0_last = &minL_r0_Array[1];
    uint16* minL_r1 = (uint16*) _mm_malloc(width*sizeof(uint16),16);
    uint16* minL_r1_last = (uint16*) _mm_malloc(width*sizeof(uint16),16);
    uint16* minL_r2 = (uint16*) _mm_malloc(width*sizeof(uint16),16);
    uint16* minL_r2_last = (uint16*) _mm_malloc(width*sizeof(uint16),16);
    uint16* minL_r3 = (uint16*) _mm_malloc(width*sizeof(uint16),16);
    uint16* minL_r3_last = (uint16*) _mm_malloc(width*sizeof(uint16),16);

    /*[formula 13 in the paper]
    compute L_r(p, d) = C(p, d) +
        min(L_r(p-r, d),
        L_r(p-r, d-1) + P1,
        L_r(p-r, d+1) + P1,
        min_k L_r(p-r, k) + P2) - min_k L_r(p-r, k)
    where p = (x,y), r is one of the directions.
        we process all the directions at once:
        ( basic 8 paths )
        0: r=(-1, 0) --> left to right
        1: r=(-1, -1) --> left to right, top to bottom
        2: r=(0, -1) --> top to bottom
        3: r=(1, -1) --> top to bottom, right to left
        ( additional ones for 16 paths )
        4: r=(-2, -1) --> two left, one down
        5: r=(-1, -1*2) --> one left, two down
        6: r=(1, -1*2) --> one right, two down
        7: r=(2, -1) --> two right, one down
    */

    // border cases L_r0[0 - disp], L_r1,2,3 is maybe not needed, as done above
    L_r0_last[-1] = L_r1_last[-1] = L_r2_last[-1] = L_r3_last[-1] = MAX_SGM_COST;
    L_r0_last[disp] = L_r1_last[disp] = L_r2_last[disp] = L_r3_last[disp] = MAX_SGM_COST;
    L_r0[-1] = L_r1[-1] =L_r2[-1]=L_r3[-1]= MAX_SGM_COST;
    L_r0[disp] = L_r1[disp] = L_r2[disp]=L_r3[disp]=MAX_SGM_COST;
    for (int pass = 0; pass < paramNoPasses; pass++) {
        int i1; int i2; int di;
        int j1; int j2; int dj;
        if (pass == 0) {
            /* top-down pass */
            i1 = 0; i2 = height; di = 1;
            j1 = 0; j2 = width;  dj = 1;
        } else {
            /* bottom-up pass */
            i1 = height-1; i2 = -1; di = -1;
            j1 = width-1; j2 = -1;  dj = -1;
        }
        img_line = img+i1*width;
        /* first line is simply costs C, except for path L_r0 */
        // first pixel
        uint16 minCost = MAX_SGM_COST;
        if (pass == 0) {
            for (int d=0; d < disp; d++) {
                uint16 cost = *getDispAddr_xyd(dsi, width, disp, i1, j1, d);
                if (cost == 255)
                    cost = paramInvalidDispCost;
                L_r0_last[d] = cost;
                L_r1_last[j1*dispP2+d] = cost;
                L_r2_last[j1*dispP2+d] = cost;
                L_r3_last[j1*dispP2+d] = cost;
                if (cost < minCost) {
                    minCost = cost;
                }
                *getDispAddr_xyd(S, width, disp, i1,j1, d) = cost;
            }
        } else {
            for (int d=0; d < disp; d++) {
                uint16 cost = *getDispAddr_xyd(dsi, width, disp, i1, j1, d);
                if (cost == 255)
                    cost = paramInvalidDispCost;
                L_r0_last[d] = cost;
                L_r1_last[j1*dispP2+d] = cost;
                L_r2_last[j1*dispP2+d] = cost;
                L_r3_last[j1*dispP2+d] = cost;
                if (cost < minCost) {
                    minCost = cost;
                }
                *getDispAddr_xyd(S, width, disp, i1,j1, d) += cost;
            }
        }
        *minL_r0_last = minCost;
        minL_r1_last[j1] = minCost;
        minL_r2_last[j1] = minCost;
        minL_r3_last[j1] = minCost;
        // rest of first line
        for (int j=j1+dj; j != j2; j += dj) {
            uint16 minCost = MAX_SGM_COST;
            *minL_r0 = MAX_SGM_COST;
            for (int d=0; d < disp; d++) {
                uint16 cost = *getDispAddr_xyd(dsi, width, disp, i1, j, d);
                if (cost == 255)
                    cost = paramInvalidDispCost;
                L_r1_last[j*dispP2+d] = cost;
                L_r2_last[j*dispP2+d] = cost;
                L_r3_last[j*dispP2+d] = cost;

                if (cost < minCost ) {
                    minCost = cost;
                }
                // minimum along L_r0
                sint32 minPropCost = L_r0_last[d]; // same disparity cost
                // P1 costs
                sint32 costP1m = L_r0_last[d-1]+paramP1;
                if (minPropCost > costP1m)
                    minPropCost = costP1m;
                sint32 costP1p = L_r0_last[d+1]+paramP1;
                if (minPropCost > costP1p) {
                    minPropCost =costP1p;
                }
                // P2 costs
                sint32 minCostP2 = *minL_r0_last;
                sint32 varP2 = adaptP2(paramAlpha, img_line[j],img_line[j-dj],paramGamma, paramP2min);
                if (minPropCost > minCostP2+varP2)
                    minPropCost = minCostP2+varP2;
                // add offset
                minPropCost -= minCostP2;
                const uint16 newCost = saturate_cast<uint16>(cost + minPropCost);
                L_r0[d] = newCost;
                if (*minL_r0 > newCost) {
                    *minL_r0 = newCost;
                }
                // cost sum
                if (pass == 0) {
                    *getDispAddr_xyd(S, width, disp, i1,j, d) = saturate_cast<uint16>(cost + minPropCost);
                }
                else{
                    *getDispAddr_xyd(S, width, disp, i1,j, d) += saturate_cast<uint16>(cost + minPropCost);
                }

            }

            minL_r1_last[j] = minCost;
            minL_r2_last[j] = minCost;
            minL_r3_last[j] = minCost;

            // swap L0 buffers
            swapPointers(L_r0, L_r0_last);
            swapPointers(minL_r0, minL_r0_last);
            // border cases: disparities -1 and disp
            L_r1_last[j*dispP2-1] = L_r2_last[j*dispP2-1] = L_r3_last[j*dispP2-1] = MAX_SGM_COST;
            L_r1_last[j*dispP2+disp] = L_r2_last[j*dispP2+disp] = L_r3_last[j*dispP2+disp] = MAX_SGM_COST;

            L_r1[j*dispP2-1] = MAX_SGM_COST;
            L_r1[j*dispP2+disp] = MAX_SGM_COST;
            L_r2[j*dispP2-1] = MAX_SGM_COST;
            L_r2[j*dispP2+disp] = MAX_SGM_COST;
            L_r3[j*dispP2-1] = MAX_SGM_COST;
            L_r3[j*dispP2+disp] = MAX_SGM_COST;
        }
        // same as img_line in first iteration, because of boundaries!
        img_line_last = img+(i1+di)*width;
        // first pixel of remaining lines
        for (int i=i1+di; i != i2; i+=di) {
            memset(L_r0_last, 0, sizeof(uint16)*disp);
            *minL_r0_last = 0;
            img_line = img+i*width;
            uint16 minCost = MAX_SGM_COST;
            minL_r2[j1] = MAX_SGM_COST;
            minL_r3[j1] = MAX_SGM_COST;
            for (int d=0; d < disp; d++) {
                uint16 cost = *getDispAddr_xyd(dsi, width, disp, i, j1, d);
                if (cost = 255 )
                    cost = paramInvalidDispCost;
                L_r0_last[d] = cost;
                L_r1_last[j1*dispP2+d] = cost;
                if (cost < minCost) {
                    minCost = cost;
                }
                //minimun along L_r2
                sint32 minPropCost_r2 = L_r2_last[j1*dispP2+d];
                //P1 costs
                sint32 costP1m_r2 = L_r2_last[j1*dispP2+d-1]+paramP1;
                if (minPropCost_r2 > costP1m_r2){
                    minPropCost_r2 = costP1m_r2;
                }
                sint32 costP1p_r2 = L_r2_last[j1*dispP2+d+1]+paramP1;
                if (minPropCost_r2 > costP1p_r2) {
                    minPropCost_r2 =costP1p_r2;
                }
                //P2 costs
                sint32 minCostP2_r2 = minL_r2_last[j1];
                sint32 varP2_r2 = adaptP2(paramAlpha, img_line[j1],img_line[j1-dj],paramGamma, paramP2min);
                if (minPropCost_r2 > minCostP2_r2 + varP2_r2)
                    minPropCost_r2 = minCostP2_r2 + varP2_r2;
                // add offset
                minPropCost_r2 -= minCostP2_r2;
                const uint16 newCost_r2 = saturate_cast<uint16>(cost + minPropCost_r2);
                L_r2[j1*dispP2+d] = newCost_r2;
                if (minL_r2[j1] > newCost_r2)
                    minL_r2[j1] = newCost_r2;
                //minimun along L_r3
                sint32 minPropCost_r3 = L_r3_last[j1*dispP2+d];
                //P1 costs
                sint32 costP1m_r3 = L_r3_last[j1*dispP2+d-1]+paramP1;
                if (minPropCost_r3 > costP1m_r3){
                    minPropCost_r3 = costP1m_r3;
                }
                sint32 costP1p_r3 = L_r3_last[j1*dispP2+d+1]+paramP1;
                if (minPropCost_r3 > costP1p_r3) {
                    minPropCost_r3 =costP1p_r3;
                }
                //P2 costs
                sint32 minCostP2_r3 = minL_r3_last[j1];
                sint32 varP2_r3 = adaptP2(paramAlpha, img_line[j1],img_line[j1-dj],paramGamma, paramP2min);
                if (minPropCost_r3 > minCostP2_r3 + varP2_r3)
                    minPropCost_r3 = minCostP2_r3 + varP2_r3;
                // add offset
                minPropCost_r3 -= minCostP2_r3;
                const uint16 newCost_r3 = saturate_cast<uint16>(cost + minPropCost_r3);
                L_r3[j1*dispP2+d] = newCost_r3;
                if (minL_r3[j1] > newCost_r3)
                    minL_r3[j1] = newCost_r3;
                const uint16 newCost =  newCost_r2 + newCost_r3;
                //cost sum
                if (pass == 0) {
                    *getDispAddr_xyd(S, width, disp, i,j1, d)= newCost;
                } else {
                    *getDispAddr_xyd(S, width, disp, i,j1, d) += newCost;
                }
            }
            *minL_r0_last = minCost;
            minL_r1[j1] = minCost;

            for (int j=j1+dj; j != j2; j += dj) {
                *minL_r0 = MAX_SGM_COST;
                minL_r1[j] = MAX_SGM_COST;
                minL_r2[j] = MAX_SGM_COST;
                minL_r3[j] = MAX_SGM_COST;
                //remaining pixels of remaining lines
                if (j!=j2-1){
                    for (int d=0; d < disp; d++) {
                        uint16 cost = *getDispAddr_xyd(dsi, width, disp, i, j, d);
                        if (cost == 255)
                            cost = paramInvalidDispCost;
                        // minimum along L_r0
                        sint32 minPropCost_r0 = L_r0_last[d]; // same disparity cost
                        // P1 costs
                        sint32 costP1m_r0 = L_r0_last[d-1]+paramP1;
                        if (minPropCost_r0 > costP1m_r0)
                            minPropCost_r0 = costP1m_r0;
                        sint32 costP1p_r0 = L_r0_last[d+1]+paramP1;
                        if (minPropCost_r0 > costP1p_r0)
                            minPropCost_r0 =costP1p_r0;

                        // P2 costs
                        sint32 minCostP2_r0 = *minL_r0_last;
                        sint32 varP2_r0 = adaptP2(paramAlpha, img_line[j],img_line[j-dj],paramGamma, paramP2min);
                        if (minPropCost_r0 > minCostP2_r0+varP2_r0)
                            minPropCost_r0 = minCostP2_r0+varP2_r0;
                        // add offset
                        minPropCost_r0 -= minCostP2_r0;
                        const uint16 newCost_r0 = saturate_cast<uint16>(cost + minPropCost_r0);
                        L_r0[d] = newCost_r0;
                        if (*minL_r0 > newCost_r0)
                            *minL_r0 = newCost_r0;
                        //minimun along L_r1
                        sint32 minPropCost_r1 = L_r1_last[j*dispP2+d];
                        // P1 costs
                        sint32 costP1m_r1 = L_r1_last[j*dispP2+d-1]+paramP1;
                        if (minPropCost_r1 > costP1m_r1)
                            minPropCost_r1 = costP1m_r1;
                        sint32 costP1p_r1 = L_r1_last[j*dispP2+d+1]+paramP1;
                        if (minPropCost_r1 > costP1p_r1)
                            minPropCost_r1 =costP1p_r1;

                        // P2 costs
                        sint32 minCostP2_r1 = minL_r1_last[j];
                        sint32 varP2_r1 = adaptP2(paramAlpha, img_line[j],img_line[j-dj],paramGamma, paramP2min);
                        if (minPropCost_r1 > minCostP2_r1+varP2_r1)
                            minPropCost_r1 = minCostP2_r1+varP2_r1;
                        // add offset
                        minPropCost_r1 -= minCostP2_r1;
                        const uint16 newCost_r1 = saturate_cast<uint16>(cost + minPropCost_r1);
                        L_r1[j*dispP2+d] = newCost_r1;
                        if (minL_r1[j] > newCost_r1)
                            minL_r1[j] = newCost_r1;
                        //minimun along L_r2
                        sint32 minPropCost_r2 = L_r2_last[j*dispP2+d];
                        //P1 costs
                        sint32 costP1m_r2 = L_r2_last[j*dispP2+d-1]+paramP1;
                        if (minPropCost_r2 > costP1m_r2)
                            minPropCost_r2 = costP1m_r2;

                        sint32 costP1p_r2 = L_r2_last[j*dispP2+d+1]+paramP1;
                        if (minPropCost_r2 > costP1p_r2)
                            minPropCost_r2 =costP1p_r2;

                        //P2 costs
                        sint32 minCostP2_r2 = minL_r2_last[j];
                        sint32 varP2_r2 = adaptP2(paramAlpha, img_line[j],img_line[j-dj],paramGamma, paramP2min);
                        if (minPropCost_r2 > minCostP2_r2 + varP2_r2)
                            minPropCost_r2 = minCostP2_r2 + varP2_r2;
                        // add offset
                        minPropCost_r2 -= minCostP2_r2;
                        const uint16 newCost_r2 = saturate_cast<uint16>(cost + minPropCost_r2);
                        L_r2[j*dispP2+d] = newCost_r2;
                        if (minL_r2[j] > newCost_r2)
                            minL_r2[j] = newCost_r2;
                        //minimun along L_r3
                        sint32 minPropCost_r3 = L_r3_last[j*dispP2+d];
                        //P1 costs
                        sint32 costP1m_r3 = L_r3_last[j*dispP2+d-1]+paramP1;
                        if (minPropCost_r3 > costP1m_r3)
                            minPropCost_r3 = costP1m_r3;

                        sint32 costP1p_r3 = L_r3_last[j*dispP2+d+1]+paramP1;
                        if (minPropCost_r3 > costP1p_r3)
                            minPropCost_r3 =costP1p_r3;

                        //P2 costs
                        sint32 minCostP2_r3 = minL_r3_last[j];
                        sint32 varP2_r3 = adaptP2(paramAlpha, img_line[j],img_line[j-dj],paramGamma, paramP2min);
                        if (minPropCost_r3 > minCostP2_r3 + varP2_r3)
                            minPropCost_r3 = minCostP2_r3 + varP2_r3;
                        // add offset
                        minPropCost_r3 -= minCostP2_r3;
                        const uint16 newCost_r3 = saturate_cast<uint16>(cost + minPropCost_r3);
                        L_r3[j*dispP2+d] = newCost_r3;
                        if (minL_r3[j] > newCost_r3)
                            minL_r3[j] = newCost_r3;
                        const uint16 newCost = newCost_r0 + newCost_r1 + newCost_r2 + newCost_r3;
                        //cost sum
                        if (pass == 0) {
                            *getDispAddr_xyd(S, width, disp, i,j, d)= newCost;
                        } else {
                            *getDispAddr_xyd(S, width, disp, i,j, d) += newCost;
                        }
                    }
                    // swap L0 buffers
                    swapPointers(L_r0, L_r0_last);
                    swapPointers(minL_r0, minL_r0_last);
                }
                //last pixels of remaining lines
                else {
                    for (int d=0; d < disp; d++){
                        uint16 cost = *getDispAddr_xyd(dsi, width, disp, i, j2-1, d);
                        if (cost == 255)
                            cost = paramInvalidDispCost;
                        // minimum along L_r0
                        sint32 minPropCost_r0 = L_r0_last[d]; // same disparity cost
                        // P1 costs
                        sint32 costP1m_r0 = L_r0_last[d-1]+paramP1;
                        if (minPropCost_r0 > costP1m_r0)
                            minPropCost_r0 = costP1m_r0;
                        sint32 costP1p_r0 = L_r0_last[d+1]+paramP1;
                        if (minPropCost_r0 > costP1p_r0)
                            minPropCost_r0 =costP1p_r0;

                        // P2 costs
                        sint32 minCostP2_r0 = *minL_r0_last;
                        sint32 varP2_r0 = adaptP2(paramAlpha, img_line[j2-1],img_line[j2-1-dj],paramGamma, paramP2min);
                        if (minPropCost_r0 > minCostP2_r0+varP2_r0)
                            minPropCost_r0 = minCostP2_r0+varP2_r0;
                        // add offset
                        minPropCost_r0 -= minCostP2_r0;
                        const uint16 newCost_r0 = saturate_cast<uint16>(cost + minPropCost_r0);
                        L_r0[d] = newCost_r0;
                        if (*minL_r0 > newCost_r0)
                            *minL_r0 = newCost_r0;
                        //minimun along L_r1
                        sint32 minPropCost_r1 = L_r1_last[(j2-1)*dispP2+d];
                        // P1 costs
                        sint32 costP1m_r1 = L_r1_last[(j2-1)*dispP2+d-1]+paramP1;
                        if (minPropCost_r1 > costP1m_r1)
                            minPropCost_r1 = costP1m_r1;
                        sint32 costP1p_r1 = L_r1_last[(j2-1)*dispP2+d+1]+paramP1;
                        if (minPropCost_r1 > costP1p_r1) {
                            minPropCost_r1 =costP1p_r1;
                        }
                        // P2 costs
                        sint32 minCostP2_r1 = minL_r1_last[j2-1];
                        sint32 varP2_r1 = adaptP2(paramAlpha, img_line[j2-1],img_line[j2-1-dj],paramGamma, paramP2min);
                        if (minPropCost_r1 > minCostP2_r1+varP2_r1)
                            minPropCost_r1 = minCostP2_r1+varP2_r1;
                        // add offset
                        minPropCost_r1 -= minCostP2_r1;
                        const uint16 newCost_r1 = saturate_cast<uint16>(cost + minPropCost_r1);
                        L_r1[(j2-1)*dispP2+d] = newCost_r1;
                        if (minL_r1[j2-1] > newCost_r1)
                            minL_r1[j2-1] = newCost_r1;
                        //minimun along L_r2
                        sint32 minPropCost_r2 = L_r2_last[(j2-1)*dispP2+d];
                        //P1 costs
                        sint32 costP1m_r2 = L_r2_last[(j2-1)*dispP2+d-1]+paramP1;
                        if (minPropCost_r2 > costP1m_r2)
                            minPropCost_r2 = costP1m_r2;

                        sint32 costP1p_r2 = L_r2_last[(j2-1)*dispP2+d+1]+paramP1;
                        if (minPropCost_r2 > costP1p_r2)
                            minPropCost_r2 =costP1p_r2;

                        //P2 costs
                        sint32 minCostP2_r2 = minL_r2_last[j2-1];
                        sint32 varP2_r2 = adaptP2(paramAlpha, img_line[j2-1],img_line[j2-1-dj],paramGamma, paramP2min);
                        if (minPropCost_r2 > minCostP2_r2 + varP2_r2)
                            minPropCost_r2 = minCostP2_r2 + varP2_r2;
                        // add offset
                        minPropCost_r2 -= minCostP2_r2;
                        const uint16 newCost_r2 = saturate_cast<uint16>(cost + minPropCost_r2);
                        L_r2[(j2-1)*dispP2+d] = newCost_r2;
                        if (minL_r2[j2-1] > newCost_r2)
                            minL_r2[j2-1] = newCost_r2;

                        const uint16 newCost = newCost_r0 + newCost_r1 + newCost_r2;
                        //cost sum
                        if (pass == 0) {
                            *getDispAddr_xyd(S, width, disp, i,j2-1, d)= newCost;
                        } else {
                            *getDispAddr_xyd(S, width, disp, i,j2-1, d) += newCost;
                        }
                    }
                }
            }

            //swap L1,L2,L3 buffers
            img_line_last = img_line;
            swapPointers(L_r1, L_r1_last);
            swapPointers(minL_r1, minL_r1_last);
            swapPointers(L_r2, L_r2_last);
            swapPointers(minL_r2, minL_r2_last);
            swapPointers(L_r3, L_r3_last);
            swapPointers(minL_r3, minL_r3_last);

        }
    }

    /* free all */
    _mm_free(L_r0-1);
    _mm_free(L_r0_last-1);
    _mm_free(L_r1-1);
    _mm_free(L_r1_last-1);
    _mm_free(L_r2-1);
    _mm_free(L_r2_last-1);
    _mm_free(L_r3-1);
    _mm_free(L_r3_last-1);
    _mm_free(minL_r1);
    _mm_free(minL_r1_last);
    _mm_free(minL_r2);
    _mm_free(minL_r2_last);
    _mm_free(minL_r3);
    _mm_free(minL_r3_last);

}
