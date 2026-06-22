/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */

/*
 * Standalone unit test for the TWILL schedule core (the twill_* helpers, which
 * live in libpico/libpico_utils.h). It makes no MPI calls and runs as a single
 * process (no mpirun); it is compiled with mpicc only because libpico_utils.h
 * includes mpi.h for its other helpers.
 *
 *   make -C utils test-twill
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libpico_utils.h"  /* TWILL schedule core lives here (with the other helpers) */

static int g_fail = 0;
static int g_checks = 0;

#define CHECK(cond, ...) do {                       \
    g_checks++;                                     \
    if (!(cond)) {                                  \
      printf("    [FAIL] ");                        \
      printf(__VA_ARGS__);                          \
      printf("\n");                                 \
      g_fail++;                                     \
    }                                               \
  } while (0)

static long labs_l(long x) { return x < 0 ? -x : x; }

/* rho is a bijection of [0,P) and rho_inv is its exact inverse. */
static void check_permutation(const int *rho, const int *rho_inv, int P,
                              const char *tag) {
  char *seen = (char *)calloc((size_t)P, 1);
  for (int i = 0; i < P; i++) {
    CHECK(rho[i] >= 0 && rho[i] < P, "%s: rho[%d]=%d out of range", tag, i, rho[i]);
    if (rho[i] >= 0 && rho[i] < P) {
      CHECK(!seen[rho[i]], "%s: rho not injective at pos %d (rank %d repeats)", tag, i, rho[i]);
      seen[rho[i]] = 1;
    }
  }
  for (int r = 0; r < P; r++) {
    CHECK(rho_inv[r] >= 0 && rho_inv[r] < P, "%s: rho_inv[%d]=%d out of range", tag, r, rho_inv[r]);
    if (rho_inv[r] >= 0 && rho_inv[r] < P) {
      CHECK(rho[rho_inv[r]] == r, "%s: rho[rho_inv[%d]] != %d", tag, r, r);
    }
  }
  free(seen);
}

/*
 * For every step t: s -> pi(s,t) is a perfect matching, and sigma is its exact
 * inverse (sigma(pi(s,t),t) == s and pi(sigma(r,t),t) == r). t==0 is the self
 * block. O(P^2); caller bounds P for the exhaustive form.
 */
static void check_inverse_and_matching(const int *rho, const int *rho_inv, int P,
                                        const char *tag) {
  char *seen = (char *)calloc((size_t)P, 1);
  for (int t = 0; t < P; t++) {
    memset(seen, 0, (size_t)P);
    for (int s = 0; s < P; s++) {
      int r = twill_pi(rho, rho_inv, P, s, t);
      if (r < 0 || r >= P) { CHECK(0, "%s: pi(%d,%d)=%d oob", tag, s, t, r); continue; }
      CHECK(!seen[r], "%s: pi not a matching at t=%d (dest %d repeats)", tag, t, r);
      seen[r] = 1;
      int s2 = twill_sigma(rho, rho_inv, P, r, t);
      CHECK(s2 == s, "%s: sigma(pi(%d,%d),%d)=%d != %d", tag, s, t, t, s2, s);
    }
  }
  for (int s = 0; s < P; s++) {
    CHECK(twill_pi(rho, rho_inv, P, s, 0) == s, "%s: pi(%d,0) != %d (self block)", tag, s, s);
    CHECK(twill_sigma(rho, rho_inv, P, s, 0) == s, "%s: sigma(%d,0) != %d (self block)", tag, s, s);
  }
  free(seen);
}

/*
 * Smooth-WRR balance for the group variant:
 *  - prefix balance (the rigorous guarantee): for every prefix length k and
 *    group j, |count_j - k*w_j/P| < 1, checked exactly as |count*P - k*w| < P.
 *  - window balance: for every window [a,b), |count_j - len*w_j/P| <= 2,
 *    which follows from the prefix bound at both ends. Window scan is bounded
 *    by P for memory/time.
 */
static void check_group_balance(const int *rho, const int *grp, int P, int G,
                                const char *tag) {
  int *w = (int *)calloc((size_t)G, sizeof(int));
  for (int r = 0; r < P; r++) { w[grp[r]]++; }

  int *cnt = (int *)calloc((size_t)G, sizeof(int));
  long worst_prefix = 0;   /* max |count*P - k*w| */
  for (int k = 1; k <= P; k++) {
    cnt[grp[rho[k - 1]]]++;
    for (int j = 0; j < G; j++) {
      long lhs = (long)cnt[j] * (long)P;
      long rhs = (long)k * (long)w[j];
      long dev = labs_l(lhs - rhs);
      if (dev > worst_prefix) { worst_prefix = dev; }
      CHECK(dev < (long)P,
            "%s: prefix balance k=%d grp=%d cnt=%d ideal=%.4f (|.|*P=%ld >= P=%d)",
            tag, k, j, cnt[j], (double)k * w[j] / P, dev, P);
    }
  }

  if ((long)(P + 1) * G <= 4000000L) {  /* bound the windowed scan */
    int *pref = (int *)calloc((size_t)(P + 1) * G, sizeof(int));
    for (int pos = 0; pos < P; pos++) {
      int g = grp[rho[pos]];
      for (int j = 0; j < G; j++) {
        pref[(pos + 1) * G + j] = pref[pos * G + j] + (j == g ? 1 : 0);
      }
    }
    long worst_window = 0;
    for (int a = 0; a < P; a++) {
      for (int b = a + 1; b <= P; b++) {
        int len = b - a;
        for (int j = 0; j < G; j++) {
          int c = pref[b * G + j] - pref[a * G + j];
          long dev = labs_l((long)c * P - (long)len * w[j]);
          if (dev > worst_window) { worst_window = dev; }
          CHECK(dev <= 2L * P,
                "%s: window [%d,%d) grp=%d cnt=%d (|.|*P=%ld > 2P=%d)",
                tag, a, b, j, c, dev, 2 * P);
        }
      }
    }
    free(pref);
    printf("    balance: worst prefix dev=%.4f, worst window dev=%.4f\n",
           (double)worst_prefix / P, (double)worst_window / P);
  } else {
    printf("    balance: worst prefix dev=%.4f (window scan skipped, P*G too large)\n",
           (double)worst_prefix / P);
  }

  free(w);
  free(cnt);
}

/* Run the full battery for one group layout (grp/P/G). */
static void run_config(const char *name, const int *grp, int P, int G) {
  printf("  config '%s' (P=%d, G=%d)\n", name, P, G);
  /* calloc (not malloc): the builders skip writing when P<=0, and zeroed
     scratch keeps the build warning-clean under -O2 inlining. */
  int *rho     = (int *)calloc((size_t)(P > 0 ? P : 1), sizeof(int));
  int *rho_inv = (int *)calloc((size_t)(P > 0 ? P : 1), sizeof(int));

  /* group variant */
  if (twill_build_rho_group(grp, P, G, rho, rho_inv) != 0) {
    CHECK(0, "%s: group rho alloc failed", name);
  } else {
    check_permutation(rho, rho_inv, P, "group");
    if (P <= 2048) { check_inverse_and_matching(rho, rho_inv, P, "group"); }
    check_group_balance(rho, grp, P, G, "group");
  }

  /* shift variant */
  twill_build_rho_shift(P, rho, rho_inv);
  check_permutation(rho, rho_inv, P, "shift");
  if (P <= 2048) { check_inverse_and_matching(rho, rho_inv, P, "shift"); }
  /* identity must map pos->pos */
  for (int i = 0; i < P; i++) { CHECK(rho[i] == i, "shift: rho[%d] != %d", i, i); }

  /* random variant */
  twill_build_rho_random(P, 0xC0FFEEULL, rho, rho_inv);
  check_permutation(rho, rho_inv, P, "random");
  if (P <= 2048) { check_inverse_and_matching(rho, rho_inv, P, "random"); }

  free(rho);
  free(rho_inv);
}

/* Build a dense grp[] from contiguous group sizes; returns P. */
static int grp_from_sizes(int *grp, const int *sizes, int G) {
  int P = 0;
  for (int j = 0; j < G; j++) {
    for (int k = 0; k < sizes[j]; k++) { grp[P++] = j; }
  }
  return P;
}

int main(void) {
  printf("== TWILL schedule unit test ==\n");

  /* ---- densify ---- */
  {
    printf("  densify\n");
    long raw[5] = {10, 10, 37, 4, 37};
    int grp[5], G = -1;
    CHECK(twill_densify(raw, 5, grp, &G) == 0, "densify returned error");
    CHECK(G == 3, "densify G=%d expected 3", G);
    int expect[5] = {1, 1, 2, 0, 2};  /* sorted unique {4,10,37} -> 0,1,2 */
    for (int i = 0; i < 5; i++) {
      CHECK(grp[i] == expect[i], "densify grp[%d]=%d expected %d", i, grp[i], expect[i]);
    }
  }

  /* ---- random determinism ---- */
  {
    printf("  random determinism\n");
    int P = 257;
    int *a = malloc(P * sizeof(int)), *ai = malloc(P * sizeof(int));
    int *b = malloc(P * sizeof(int)), *bi = malloc(P * sizeof(int));
    twill_build_rho_random(P, 0xABCDEF12345ULL, a, ai);
    twill_build_rho_random(P, 0xABCDEF12345ULL, b, bi);
    CHECK(memcmp(a, b, P * sizeof(int)) == 0, "random not deterministic for equal seed");
    twill_build_rho_random(P, 0xABCDEF12346ULL, b, bi);
    CHECK(memcmp(a, b, P * sizeof(int)) != 0, "random identical across different seeds (suspicious)");
    free(a); free(ai); free(b); free(bi);
  }

  /* ---- group layouts ---- */
  int buf[4096];

  { int s[1] = {1};            int P = grp_from_sizes(buf, s, 1); run_config("single rank [1]", buf, P, 1); }
  { int s[1] = {13};           int P = grp_from_sizes(buf, s, 1); run_config("one group [13] (prime)", buf, P, 1); }
  { int s[2] = {1, 1};         int P = grp_from_sizes(buf, s, 2); run_config("two singletons [1,1]", buf, P, 2); }
  { int s[3] = {1, 1, 5};      int P = grp_from_sizes(buf, s, 3); run_config("ragged [1,1,5]", buf, P, 3); }
  { int s[3] = {1, 1, 37};     int P = grp_from_sizes(buf, s, 3); run_config("ragged [1,1,37]", buf, P, 3); }
  { int s[2] = {1, 37};        int P = grp_from_sizes(buf, s, 2); run_config("ragged [1,37]", buf, P, 2); }
  { int s[4] = {4, 4, 4, 4};   int P = grp_from_sizes(buf, s, 4); run_config("uniform 4x4", buf, P, 4); }
  { int s[2] = {4, 4};         int P = grp_from_sizes(buf, s, 2); run_config("uniform 2x4 (P=8)", buf, P, 2); }
  { int s[5] = {3, 7, 2, 11, 5}; int P = grp_from_sizes(buf, s, 5); run_config("mixed [3,7,2,11,5]", buf, P, 5); }
  { int s[2] = {7, 6};         int P = grp_from_sizes(buf, s, 2); run_config("near-equal [7,6] (P=13 prime)", buf, P, 2); }

  /* interleaved (non-contiguous) group ids: members must still be rank-sorted */
  {
    int P = 12, G = 3;
    for (int r = 0; r < P; r++) { buf[r] = r % G; }  /* 0,1,2,0,1,2,... */
    run_config("interleaved 0,1,2,...", buf, P, G);
  }

  /* large, randomly-assigned groups */
  {
    int P = 1000, G = 16;
    uint64_t st = 0x1234ULL;
    int nonempty = 0;
    int seen_any[16] = {0};
    for (int r = 0; r < P; r++) {
      buf[r] = (int)twill_rand_bounded(&st, (uint64_t)G);
      if (!seen_any[buf[r]]) { seen_any[buf[r]] = 1; nonempty++; }
    }
    /* densify so dense ids are contiguous even if a group ended up empty */
    long *raw = malloc(P * sizeof(long));
    int *grp = malloc(P * sizeof(int));
    int Gd = 0;
    for (int r = 0; r < P; r++) { raw[r] = buf[r]; }
    twill_densify(raw, P, grp, &Gd);
    run_config("random groups P=1000 G~16", grp, P, Gd);
    free(raw); free(grp);
    (void)nonempty;
  }

  /* larger P, exhaustive matching skipped, balance windowed-scan skipped */
  {
    int s[3] = {1000, 1500, 1500};
    int *big = malloc(4000 * sizeof(int));
    int P = grp_from_sizes(big, s, 3);
    run_config("large ragged [1000,1500,1500]", big, P, 3);
    free(big);
  }

  printf("\n%d checks run, %d failures\n", g_checks, g_fail);
  if (g_fail == 0) {
    printf("ALL TESTS PASSED\n");
    return 0;
  }
  printf("TESTS FAILED\n");
  return 1;
}
