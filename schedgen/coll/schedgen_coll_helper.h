/*
 * Copyright (c) 2009 The Trustees of Indiana University and Indiana
 *                    University Research and Technology
 *                    Corporation.  All rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *            Timo Schneider <timoschn@cs.indiana.edu>
 *
 */

#ifndef SCHEDGEN_COLL_HELPER_H
#define SCHEDGEN_COLL_HELPER_H

#include <math.h>
#include <stdint.h>

class Goal;
struct gengetopt_args_info;

/**
 * Returns the nearest (smaller than or equal to) power of two of a number
 * Implementation from:
 * https://github.com/pmodels/mpich/blob/main/src/mpl/include/mpl_math.h#L35
 **/
static inline int comm_size_pof2(int number) {
  if (number > 0) {
        return 1 << (int) log2(number);
    } else {
        return 0;
    }
}

static inline int rank_to_vrank(int rank, int root) {
  if (rank == 0) {
    return root;
  }
  if (rank == root) {
    return 0;
  }
  return rank;
}

static inline int vrank_to_rank(int vrank, int root) {
  if (vrank == 0) {
    return root;
  }
  if (vrank == root) {
    return 0;
  }
  return vrank;
}

static inline int mymod(int a, int b) {
  // calculate a % b
  while (a < 0) {
    a += b;
  }
  return a % b;
}

static inline double mylog(double base, double x) { return log(x) / log(base); }

uint32_t MAKE_TAG(int comm, int tag);

#endif // SCHEDGEN_COLL_HELPER_H
