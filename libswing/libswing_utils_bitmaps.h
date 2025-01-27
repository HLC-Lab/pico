#ifndef LIBSWING_UTILS_BITMAPS_H
#define LIBSWING_UTILS_BITMAPS_H

#include <stddef.h>

extern const int send_2[2][1];
extern const int recv_2[2][1];

extern const int send_4[4][2];
extern const int recv_4[4][2];

extern const int send_8[8][3];
extern const int recv_8[8][3];

extern const int send_16[16][4];
extern const int recv_16[16][4];

extern const int send_32[32][5];
extern const int recv_32[32][5];

extern const int send_64[64][6];
extern const int recv_64[64][6];

extern const int send_128[128][7];
extern const int recv_128[128][7];

extern const int send_256[256][8];
extern const int recv_256[256][8];

extern const void* static_send_bitmaps[];
extern const void* static_recv_bitmaps[];

static inline int get_static_bitmap(const int** send_bitmap, const int** recv_bitmap, int n_steps, int comm_sz, int rank) {
  // verify that comm_sz is exactly 2^n_steps
  if (comm_sz != (1 << n_steps)){
    return -1;
  }
  // Static bitmaps are defined up to 256 ranks, so since n_steps = log2 comm_sz -> log2(256)=8
  if (n_steps < 1 || n_steps > 8) {
    return -1;
  }

  *send_bitmap = ((const int*)static_send_bitmaps[n_steps]) + (ptrdiff_t)(rank * n_steps);
  *recv_bitmap = ((const int*)static_recv_bitmaps[n_steps]) + (ptrdiff_t)(rank * n_steps);

  return 0;  // Success
}

#endif
