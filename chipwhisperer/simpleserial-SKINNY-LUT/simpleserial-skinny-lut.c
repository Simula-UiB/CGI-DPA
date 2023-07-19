/*
    This file is part of the ChipWhisperer Example Targets
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

uint8_t SBOX[256] = {
    0x65, 0x4c, 0x6a, 0x42, 0x4b, 0x63, 0x43, 0x6b, 0x55, 0x75, 0x5a, 0x7a, 0x53, 0x73, 0x5b, 0x7b,
    0x35, 0x8c, 0x3a, 0x81, 0x89, 0x33, 0x80, 0x3b, 0x95, 0x25, 0x98, 0x2a, 0x90, 0x23, 0x99, 0x2b,
    0xe5, 0xcc, 0xe8, 0xc1, 0xc9, 0xe0, 0xc0, 0xe9, 0xd5, 0xf5, 0xd8, 0xf8, 0xd0, 0xf0, 0xd9, 0xf9,
    0xa5, 0x1c, 0xa8, 0x12, 0x1b, 0xa0, 0x13, 0xa9, 0x05, 0xb5, 0x0a, 0xb8, 0x03, 0xb0, 0x0b, 0xb9,
    0x32, 0x88, 0x3c, 0x85, 0x8d, 0x34, 0x84, 0x3d, 0x91, 0x22, 0x9c, 0x2c, 0x94, 0x24, 0x9d, 0x2d,
    0x62, 0x4a, 0x6c, 0x45, 0x4d, 0x64, 0x44, 0x6d, 0x52, 0x72, 0x5c, 0x7c, 0x54, 0x74, 0x5d, 0x7d,
    0xa1, 0x1a, 0xac, 0x15, 0x1d, 0xa4, 0x14, 0xad, 0x02, 0xb1, 0x0c, 0xbc, 0x04, 0xb4, 0x0d, 0xbd,
    0xe1, 0xc8, 0xec, 0xc5, 0xcd, 0xe4, 0xc4, 0xed, 0xd1, 0xf1, 0xdc, 0xfc, 0xd4, 0xf4, 0xdd, 0xfd,
    0x36, 0x8e, 0x38, 0x82, 0x8b, 0x30, 0x83, 0x39, 0x96, 0x26, 0x9a, 0x28, 0x93, 0x20, 0x9b, 0x29,
    0x66, 0x4e, 0x68, 0x41, 0x49, 0x60, 0x40, 0x69, 0x56, 0x76, 0x58, 0x78, 0x50, 0x70, 0x59, 0x79,
    0xa6, 0x1e, 0xaa, 0x11, 0x19, 0xa3, 0x10, 0xab, 0x06, 0xb6, 0x08, 0xba, 0x00, 0xb3, 0x09, 0xbb,
    0xe6, 0xce, 0xea, 0xc2, 0xcb, 0xe3, 0xc3, 0xeb, 0xd6, 0xf6, 0xda, 0xfa, 0xd3, 0xf3, 0xdb, 0xfb,
    0x31, 0x8a, 0x3e, 0x86, 0x8f, 0x37, 0x87, 0x3f, 0x92, 0x21, 0x9e, 0x2e, 0x97, 0x27, 0x9f, 0x2f,
    0x61, 0x48, 0x6e, 0x46, 0x4f, 0x67, 0x47, 0x6f, 0x51, 0x71, 0x5e, 0x7e, 0x57, 0x77, 0x5f, 0x7f,
    0xa2, 0x18, 0xae, 0x16, 0x1f, 0xa7, 0x17, 0xaf, 0x01, 0xb2, 0x0e, 0xbe, 0x07, 0xb7, 0x0f, 0xbf,
    0xe2, 0xca, 0xee, 0xc6, 0xcf, 0xe7, 0xc7, 0xef, 0xd2, 0xf2, 0xde, 0xfe, 0xd7, 0xf7, 0xdf, 0xff};

//Adapted from FELICS framework

void encrypt(uint8_t *block, uint8_t *roundKeys, uint8_t *SBOX, uint8_t rounds)
{
    // r0     : points to plaintext
    // r1     : points to roundKeys
    // r2-r5  : cipher state
    // r6 : points to SBOX
    // r7-r10 : temp use
    // r12 : 0xff
    // r11    : loop control
    asm volatile(
        "stmdb      sp!,      {r3-r12}         \n\t"
        "mov        r6,        r2              \n\t"
        "mov        r11,       r3              \n\t"
        "ldmia      r0,       {r2-r5}          \n\t" // load plaintext
        "mov        r12,       #0xff           \n\t"
        // r2 (s3  s2  s1  s0)
        // r3 (s7  s6  s5  s4)
        // r4 (s11 s10 s9  s8)
        // r5 (s15 s14 s13 s12)
        "enc_loop:                             \n\t"
        // SubColumn
        // first row
        "and        r7,       r2, #0xff        \n\t"
        "ldrb       r7,       [r6,r7]          \n\t"
        "bfi        r2,r7,    #0, #8           \n\t"
        "and        r7,       r12, r2, lsr #8  \n\t"
        "ldrb       r7,       [r6,r7]          \n\t"
        "bfi        r2,r7,    #8, #8           \n\t"
        "and        r7,       r12, r2, lsr #16 \n\t"
        "ldrb       r7,       [r6,r7]          \n\t"
        "bfi        r2,r7,    #16, #8          \n\t"
        "mov        r7,       r2, lsr #24      \n\t"
        "ldrb       r7,       [r6,r7]          \n\t"
        "bfi        r2,r7,    #24, #8          \n\t"
        // second row
        "and        r7,       r3, #0xff        \n\t"
        "ldrb       r7,       [r6,r7]          \n\t"
        "bfi        r3,r7,    #0, #8           \n\t"
        "and        r7,       r12, r3, lsr #8  \n\t"
        "ldrb       r7,       [r6,r7]          \n\t"
        "bfi        r3,r7,    #8, #8           \n\t"
        "and        r7,       r12, r3, lsr #16 \n\t"
        "ldrb       r7,       [r6,r7]          \n\t"
        "bfi        r3,r7,    #16, #8          \n\t"
        "mov        r7,       r3, lsr #24      \n\t"
        "ldrb       r7,       [r6,r7]          \n\t"
        "bfi        r3,r7,    #24, #8          \n\t"
        // third row
        "and        r7,       r4, #0xff        \n\t"
        "ldrb       r7,       [r6,r7]          \n\t"
        "bfi        r4,r7,    #0, #8           \n\t"
        "and        r7,       r12, r4, lsr #8  \n\t"
        "ldrb       r7,       [r6,r7]          \n\t"
        "bfi        r4,r7,    #8, #8           \n\t"
        "and        r7,       r12, r4, lsr #16 \n\t"
        "ldrb       r7,       [r6,r7]          \n\t"
        "bfi        r4,r7,    #16, #8          \n\t"
        "mov        r7,       r4, lsr #24      \n\t"
        "ldrb       r7,       [r6,r7]          \n\t"
        "bfi        r4,r7,    #24, #8          \n\t"
        // fourth row
        "and        r7,       r5, #0xff        \n\t"
        "ldrb       r7,       [r6,r7]          \n\t"
        "bfi        r5,r7,    #0, #8           \n\t"
        "and        r7,       r12, r5, lsr #8  \n\t"
        "ldrb       r7,       [r6,r7]          \n\t"
        "bfi        r5,r7,    #8, #8           \n\t"
        "and        r7,       r12, r5, lsr #16 \n\t"
        "ldrb       r7,       [r6,r7]          \n\t"
        "bfi        r5,r7,    #16, #8          \n\t"
        "mov        r7,       r5, lsr #24      \n\t"
        "ldrb       r7,       [r6,r7]          \n\t"
        "bfi        r5,r7,    #24, #8          \n\t"
        // AddRoundKey and AddRoundConst
        // Note that the rounds keys already contain
        // C0 and C1 so we only add C2 to the third row
        "ldrd       r7,r8,    [r1,#0]          \n\t"
        "adds       r1,       r1, #8           \n\t"
        "eors       r2,       r2, r7           \n\t"
        "eors       r3,       r3, r8           \n\t"
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
        "mov        r8,       r2               \n\t"
        "mov        r2,       r5               \n\t"
        "mov        r5,       r4               \n\t"
        "mov        r4,       r3               \n\t"
        "mov        r3,       r8               \n\t"
        "subs       r11,      r11, #1          \n\t"
        "bne        enc_loop                   \n\t"
        "stmia      r0,       {r2-r5}          \n\t" // store back ciphertext in block
        "ldmia      sp!,      {r3-r12}         \n\t"
        :
        : [block] "r"(block), [roundKeys] "r"(roundKeys), [SBOX] "r"(SBOX), [rounds] "r"(rounds));
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
        encrypt(pt, keys, SBOX, trace_start);
    // recording traces
    trigger_high();

    encrypt(pt, keys + trace_start * 8, SBOX, 56 - trace_start);

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
    trace_start = 47;
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
