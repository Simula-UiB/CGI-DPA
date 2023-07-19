/*
    This file was made using a file of the ChipWhisperer Example Targets
    Copyright (C) 2012-2017 NewAE Technology Inc.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "hal.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "simpleserial.h"

uint8_t keys[448];
uint32_t tk1[4];
uint32_t tk2[4];
uint8_t trace_start = 0;

void encrypt(uint8_t *block, uint8_t *roundKeys, uint8_t rounds)
{
    // r0     : points to plaintext
    // r1     : points to roundKeys
    // r2-r5  : cipher state
    // r6-r8,r12 : temp use
    // r9-r10 : circuit constants
    // r11    : loop control
    asm volatile(
        "stmdb      sp!,      {r2-r12}         \n\t"
        "mov        r11,       r2              \n\t"
        "ldmia      r0,       {r2-r5}          \n\t" // load plaintext
        "ldr     r9, =0x4040404                \n\t" // circuit constant
        "ldr     r10, =0x20202020              \n\t" // circuit constant
        // r2 (s3  s2  s1  s0)
        // r3 (s7  s6  s5  s4)
        // r4 (s11 s10 s9  s8)
        // r5 (s15 s14 s13 s12)
        "enc_loop:                             \n\t"
        // SubColumn
        // original FELICS version used a LUT
        // here we use the circuit from rweather https://github.com/rweather/skinny-c
        // described in skinny128-cipher, function sbox for 32 bits
        // it allows for computing the sbox on an entire row in constant time

        // first row
        "mvn     r6, r2                        \n\t"
        "ldr     r2, =0x40404040               \n\t"
        "ldr     r8, =0x11111111               \n\t"
        "and     r7, r2, r6, lsl #4            \n\t"
        "lsr     r2, r6, #3                    \n\t"
        "and     r2, r2, r6, lsr #2            \n\t"
        "and     r2, r2, r8                    \n\t"
        "eor     r8, r2, r6                    \n\t"
        "lsr     r6, r6, #2                    \n\t"
        "and     r2, r10, r8, lsl #1           \n\t"
        "orr     r2, r2, r7                    \n\t"
        "and     r2, r2, r8, lsl #5            \n\t"
        "eor     r2, r2, r8                    \n\t"
        "lsr     r8, r8, #1                    \n\t"
        "lsl     r7, r2, #1                    \n\t"
        "and     r6, r6, r2, lsl #1            \n\t"
        "and     r12, r7, r2, lsl #2           \n\t"
        "ldr     r7, =0x80808080               \n\t"
        "and     r7, r12, r7                   \n\t"
        "ldr     r12, =0x2020202               \n\t"
        "and     r6, r6, r12                   \n\t"
        "orr     r6, r7, r6                    \n\t"
        "eor     r6, r6, r2                    \n\t"
        "and     r2, r8, r2, lsr #2            \n\t"
        "ldr     r8, =0x8080808                \n\t"
        "lsl     r7, r6, #1                    \n\t"
        "and     r7, r7, r6, lsr #5            \n\t"
        "mvn     r6, r6                        \n\t"
        "and     r7, r7, r9                    \n\t"
        "and     r2, r2, r8                    \n\t"
        "ldr     r8, =0x10101010               \n\t"
        "orr     r2, r7, r2                    \n\t"
        "ldr     r7, =0x1010101                \n\t"
        "eor     r2, r2, r6                    \n\t"
        "and     r7, r7, r2, lsr #2            \n\t"
        "and     r2, r8, r2, lsl #1            \n\t"
        "ldr     r8, =0xc8c8c8c8               \n\t"
        "and     r8, r8, r6, lsl #2            \n\t"
        "orr     r2, r2, r8                    \n\t"
        "and     r8, r10, r6, lsl #5           \n\t"
        "orr     r2, r2, r8                    \n\t"
        "and     r8, r12, r6, lsr #6           \n\t"
        "and     r6, r9, r6, lsr #4            \n\t"
        "orr     r2, r2, r8                    \n\t"
        "orr     r2, r2, r6                    \n\t"
        "orr     r2, r2, r7                    \n\t"
        // second row
        "mvn     r6, r3                        \n\t"
        "ldr     r3, =0x40404040               \n\t"
        "ldr     r8, =0x11111111               \n\t"
        "and     r7, r3, r6, lsl #4            \n\t"
        "lsr     r3, r6, #3                    \n\t"
        "and     r3, r3, r6, lsr #2            \n\t"
        "and     r3, r3, r8                    \n\t"
        "eor     r8, r3, r6                    \n\t"
        "lsr     r6, r6, #2                    \n\t"
        "and     r3, r10, r8, lsl #1           \n\t"
        "orr     r3, r3, r7                    \n\t"
        "and     r3, r3, r8, lsl #5            \n\t"
        "eor     r3, r3, r8                    \n\t"
        "lsr     r8, r8, #1                    \n\t"
        "lsl     r7, r3, #1                    \n\t"
        "and     r6, r6, r3, lsl #1            \n\t"
        "and     r12, r7, r3, lsl #2           \n\t"
        "ldr     r7, =0x80808080               \n\t"
        "and     r7, r12, r7                   \n\t"
        "ldr     r12, =0x2020202               \n\t"
        "and     r6, r6, r12                   \n\t"
        "orr     r6, r7, r6                    \n\t"
        "eor     r6, r6, r3                    \n\t"
        "and     r3, r8, r3, lsr #2            \n\t"
        "ldr     r8, =0x8080808                \n\t"
        "lsl     r7, r6, #1                    \n\t"
        "and     r7, r7, r6, lsr #5            \n\t"
        "mvn     r6, r6                        \n\t"
        "and     r7, r7, r9                    \n\t"
        "and     r3, r3, r8                    \n\t"
        "ldr     r8, =0x10101010               \n\t"
        "orr     r3, r7, r3                    \n\t"
        "ldr     r7, =0x1010101                \n\t"
        "eor     r3, r3, r6                    \n\t"
        "and     r7, r7, r3, lsr #2            \n\t"
        "and     r3, r8, r3, lsl #1            \n\t"
        "ldr     r8, =0xc8c8c8c8               \n\t"
        "and     r8, r8, r6, lsl #2            \n\t"
        "orr     r3, r3, r8                    \n\t"
        "and     r8, r10, r6, lsl #5           \n\t"
        "orr     r3, r3, r8                    \n\t"
        "and     r8, r12, r6, lsr #6           \n\t"
        "and     r6, r9, r6, lsr #4            \n\t"
        "orr     r3, r3, r8                    \n\t"
        "orr     r3, r3, r6                    \n\t"
        "orr     r3, r3, r7                    \n\t"
        // third row
        "mvn     r6, r4                        \n\t"
        "ldr     r4, =0x40404040               \n\t"
        "ldr     r8, =0x11111111               \n\t"
        "and     r7, r4, r6, lsl #4            \n\t"
        "lsr     r4, r6, #3                    \n\t"
        "and     r4, r4, r6, lsr #2            \n\t"
        "and     r4, r4, r8                    \n\t"
        "eor     r8, r4, r6                    \n\t"
        "lsr     r6, r6, #2                    \n\t"
        "and     r4, r10, r8, lsl #1           \n\t"
        "orr     r4, r4, r7                    \n\t"
        "and     r4, r4, r8, lsl #5            \n\t"
        "eor     r4, r4, r8                    \n\t"
        "lsr     r8, r8, #1                    \n\t"
        "lsl     r7, r4, #1                    \n\t"
        "and     r6, r6, r4, lsl #1            \n\t"
        "and     r12, r7, r4, lsl #2           \n\t"
        "ldr     r7, =0x80808080               \n\t"
        "and     r7, r12, r7                   \n\t"
        "ldr     r12, =0x2020202               \n\t"
        "and     r6, r6, r12                   \n\t"
        "orr     r6, r7, r6                    \n\t"
        "eor     r6, r6, r4                    \n\t"
        "and     r4, r8, r4, lsr #2            \n\t"
        "ldr     r8, =0x8080808                \n\t"
        "lsl     r7, r6, #1                    \n\t"
        "and     r7, r7, r6, lsr #5            \n\t"
        "mvn     r6, r6                        \n\t"
        "and     r7, r7, r9                    \n\t"
        "and     r4, r4, r8                    \n\t"
        "ldr     r8, =0x10101010               \n\t"
        "orr     r4, r7, r4                    \n\t"
        "ldr     r7, =0x1010101                \n\t"
        "eor     r4, r4, r6                    \n\t"
        "and     r7, r7, r4, lsr #2            \n\t"
        "and     r4, r8, r4, lsl #1            \n\t"
        "ldr     r8, =0xc8c8c8c8               \n\t"
        "and     r8, r8, r6, lsl #2            \n\t"
        "orr     r4, r4, r8                    \n\t"
        "and     r8, r10, r6, lsl #5           \n\t"
        "orr     r4, r4, r8                    \n\t"
        "and     r8, r12, r6, lsr #6           \n\t"
        "and     r6, r9, r6, lsr #4            \n\t"
        "orr     r4, r4, r8                    \n\t"
        "orr     r4, r4, r6                    \n\t"
        "orr     r4, r4, r7                    \n\t"
        // fourth row
        "mvn     r6, r5                        \n\t"
        "ldr     r5, =0x40404040               \n\t"
        "ldr     r8, =0x11111111               \n\t"
        "and     r7, r5, r6, lsl #4            \n\t"
        "lsr     r5, r6, #3                    \n\t"
        "and     r5, r5, r6, lsr #2            \n\t"
        "and     r5, r5, r8                    \n\t"
        "eor     r8, r5, r6                    \n\t"
        "lsr     r6, r6, #2                    \n\t"
        "and     r5, r10, r8, lsl #1           \n\t"
        "orr     r5, r5, r7                    \n\t"
        "and     r5, r5, r8, lsl #5            \n\t"
        "eor     r5, r5, r8                    \n\t"
        "lsr     r8, r8, #1                    \n\t"
        "lsl     r7, r5, #1                    \n\t"
        "and     r6, r6, r5, lsl #1            \n\t"
        "and     r12, r7, r5, lsl #2           \n\t"
        "ldr     r7, =0x80808080               \n\t"
        "and     r7, r12, r7                   \n\t"
        "ldr     r12, =0x2020202               \n\t"
        "and     r6, r6, r12                   \n\t"
        "orr     r6, r7, r6                    \n\t"
        "eor     r6, r6, r5                    \n\t"
        "and     r5, r8, r5, lsr #2            \n\t"
        "ldr     r8, =0x8080808                \n\t"
        "lsl     r7, r6, #1                    \n\t"
        "and     r7, r7, r6, lsr #5            \n\t"
        "mvn     r6, r6                        \n\t"
        "and     r7, r7, r9                    \n\t"
        "and     r5, r5, r8                    \n\t"
        "ldr     r8, =0x10101010               \n\t"
        "orr     r5, r7, r5                    \n\t"
        "ldr     r7, =0x1010101                \n\t"
        "eor     r5, r5, r6                    \n\t"
        "and     r7, r7, r5, lsr #2            \n\t"
        "and     r5, r8, r5, lsl #1            \n\t"
        "ldr     r8, =0xc8c8c8c8               \n\t"
        "and     r8, r8, r6, lsl #2            \n\t"
        "orr     r5, r5, r8                    \n\t"
        "and     r8, r10, r6, lsl #5           \n\t"
        "orr     r5, r5, r8                    \n\t"
        "and     r8, r12, r6, lsr #6           \n\t"
        "and     r6, r9, r6, lsr #4            \n\t"
        "orr     r5, r5, r8                    \n\t"
        "orr     r5, r5, r6                    \n\t"
        "orr     r5, r5, r7                    \n\t"
        // AddRoundKey and AddRoundConst
        // Note that the rounds keys already contain
        // C0 and C1 so we only add C2 to the third row
        "ldrd       r6,r7,    [r1,#0]          \n\t"
        "adds       r1,       r1, #8           \n\t"
        "eors       r2,       r2, r6           \n\t"
        "eors       r3,       r3, r7           \n\t"
        "eors       r4,       r4, #0x02        \n\t"
        // ShiftRow
        // Opposite of the spec because of little endianess
        "rors       r3,       r3, #24          \n\t"
        "rors       r4,       r4, #16          \n\t"
        "rors       r5,       r5, #8           \n\t"
        // MixColumn
        "eors       r3,       r3, r4           \n\t"
        "eors       r4,       r4, r2           \n\t"
        "eors       r5,       r5, r4           \n\t"
        "mov        r6,       r2               \n\t"
        "mov        r2,       r5               \n\t"
        "mov        r5,       r4               \n\t"
        "mov        r4,       r3               \n\t"
        "mov        r3,       r6               \n\t"
        "subs       r11,      r11, #1          \n\t"
        "bne        enc_loop                   \n\t"
        "stmia      r0,       {r2-r5}          \n\t" // store back ciphertext in block
        "ldmia      sp!,      {r2-r12}         \n\t"
        :
        : [block] "r"(block), [roundKeys] "r"(roundKeys), [rounds] "r"(rounds));
}

uint8_t get_tk1(uint8_t *k)
{
    uint32_t tk1_raw[4] = {k[0] | (k[1] << 8) | (k[2] << 16) | (k[3] << 24), k[4] | (k[5] << 8) | (k[6] << 16) | (k[7] << 24),
                           k[8] | (k[9] << 8) | (k[10] << 16) | (k[11] << 24), k[12] | (k[13] << 8) | (k[14] << 16) | (k[15] << 24)};
    memcpy(tk1, tk1_raw, sizeof(tk1_raw));
    return 0x00;
}

uint8_t get_tk2(uint8_t *k)
{
    uint32_t tk2_raw[4] = {k[0] | (k[1] << 8) | (k[2] << 16) | (k[3] << 24), k[4] | (k[5] << 8) | (k[6] << 16) | (k[7] << 24),
                           k[8] | (k[9] << 8) | (k[10] << 16) | (k[11] << 24), k[12] | (k[13] << 8) | (k[14] << 16) | (k[15] << 24)};
    memcpy(tk2, tk2_raw, sizeof(tk2_raw));
    return 0x00;
}


// NOTE: get_key effectively destroys tk1 and tk2
// When providing a new key, tk1 and tk2 need to be
// provided again even if they stay constant
uint8_t get_key(uint8_t *k)
{
    // result should be a uint8_t array stored in keys
    // where each round key is the top two rows of each TK xored together
    // and already contains the rounds constants c0 and c1
    uint32_t tk3[4] = {k[0] | (k[1] << 8) | (k[2] << 16) | (k[3] << 24), k[4] | (k[5] << 8) | (k[6] << 16) | (k[7] << 24),
                       k[8] | (k[9] << 8) | (k[10] << 16) | (k[11] << 24), k[12] | (k[13] << 8) | (k[14] << 16) | (k[15] << 24)};
    uint8_t c0[56] = {1, 3, 7, 15, 15, 14, 13, 11, 7, 15, 14, 12, 9, 3, 7, 14, 13, 10, 5, 11, 6, 12, 8, 0, 1, 2, 5, 11, 7, 14, 12, 8,
                      1, 3, 6, 13, 11, 6, 13, 10, 4, 9, 2, 4, 8, 1, 2, 4, 9, 3, 6, 12, 9, 2, 5, 10};
    uint8_t c1[56] = {0, 0, 0, 0, 1, 3, 3, 3, 3, 2, 1, 3, 3, 3, 2, 0, 1, 3, 3, 2, 1, 2, 1, 3, 2, 0, 0, 0, 1, 2, 1, 3, 3, 2, 0, 0, 1,
                      3, 2, 1, 3, 2, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0, 1, 3, 2, 0};
    int i;
    for (i = 0; i < 56; i++)
    {
        uint32_t rkey = tk1[0] ^ tk2[0] ^ tk3[0] ^ c0[i];
        memcpy(keys + i * 8, &rkey, 4);
        rkey = tk1[1] ^ tk2[1] ^ tk3[1] ^ c1[i];
        memcpy(keys + i * 8 + 4, &rkey, 4);
        // Permutation taken from skinny c by rweather https://github.com/rweather/skinny-c
        //tk1
        uint32_t row2 = tk1[2];
        uint32_t row3 = tk1[3];
        tk1[2] = tk1[0];
        tk1[3] = tk1[1];
        row3 = (row3 << 16) | (row3 >> 16);
        tk1[0] = ((row2 >> 8) & 0x000000FFU) |
                 ((row2 << 16) & 0x00FF0000U) |
                 (row3 & 0xFF00FF00U);
        tk1[1] = ((row2 >> 16) & 0x000000FFU) |
                 (row2 & 0xFF000000U) |
                 ((row3 << 8) & 0x0000FF00U) |
                 (row3 & 0x00FF0000U);
        //tk2
        row2 = tk2[2];
        row3 = tk2[3];
        tk2[2] = tk2[0];
        tk2[3] = tk2[1];
        row3 = (row3 << 16) | (row3 >> 16);
        tk2[0] = ((row2 >> 8) & 0x000000FFU) |
                 ((row2 << 16) & 0x00FF0000U) |
                 (row3 & 0xFF00FF00U);
        tk2[1] = ((row2 >> 16) & 0x000000FFU) |
                 (row2 & 0xFF000000U) |
                 ((row3 << 8) & 0x0000FF00U) |
                 (row3 & 0x00FF0000U);
        // LFSR2 taken from skinny c by rweather https://github.com/rweather/skinny-c
        tk2[0] = ((tk2[0] << 1) & 0xFEFEFEFEU) ^ (((tk2[0] >> 7) ^ (tk2[0] >> 5)) & 0x01010101U);
        tk2[1] = ((tk2[1] << 1) & 0xFEFEFEFEU) ^ (((tk2[1] >> 7) ^ (tk2[1] >> 5)) & 0x01010101U);
        //tk3
        row2 = tk3[2];
        row3 = tk3[3];
        tk3[2] = tk3[0];
        tk3[3] = tk3[1];
        row3 = (row3 << 16) | (row3 >> 16);
        tk3[0] = ((row2 >> 8) & 0x000000FFU) |
                 ((row2 << 16) & 0x00FF0000U) |
                 (row3 & 0xFF00FF00U);
        tk3[1] = ((row2 >> 16) & 0x000000FFU) |
                 (row2 & 0xFF000000U) |
                 ((row3 << 8) & 0x0000FF00U) |
                 (row3 & 0x00FF0000U);
        // LFSR3 taken from skinny c by rweather https://github.com/rweather/skinny-c
        tk3[0] = ((tk3[0] >> 1) & 0x7F7F7F7FU) ^ (((tk3[0] << 7) ^ (tk3[0] << 1)) & 0x80808080U);
        tk3[1] = ((tk3[1] >> 1) & 0x7F7F7F7FU) ^ (((tk3[1] << 7) ^ (tk3[1] << 1)) & 0x80808080U);
    }
    return 0x00;
}

uint8_t get_pt(uint8_t *pt)
{
    if (trace_start > 0)
        encrypt(pt, keys, trace_start);

    // recording traces
    trigger_high();

    encrypt(pt, keys + trace_start * 8, 56 - trace_start);

    trigger_low();
    simpleserial_put('r', 16, pt);
    // write back ciphertext
    return 0x00;
}

uint8_t set_start(uint8_t *pt)
{
    trace_start = 0;
    return 0x00;
}

uint8_t set_end(uint8_t *pt)
{
    trace_start = 51;
    return 0x00;
}

uint8_t reset(uint8_t *x)
{
    // Reset key here if needed
    return 0x00;
}

int main(void)
{
    platform_init();
    init_uart();
    trigger_setup();

    simpleserial_init();
    simpleserial_addcmd('1', 16, get_tk1);
    simpleserial_addcmd('2', 16, get_tk2);
    simpleserial_addcmd('k', 16, get_key);
    simpleserial_addcmd('p', 16, get_pt);
    simpleserial_addcmd('s', 0, set_start);
    simpleserial_addcmd('e', 0, set_end);
    simpleserial_addcmd('x', 0, reset);
    while (1)
        simpleserial_get();
}
