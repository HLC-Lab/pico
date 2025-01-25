#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "test_selection.h"

static inline int set_allreduce_algorithm(allreduce_algo_t * algorithm, int algo_number){
  switch (algo_number){
    case 0:
      *algorithm = ALLREDUCE_DEFAULT;
      break;
    case 1:
      *algorithm = ALLREDUCE_LINEAR;
      break;
    case 2:
      *algorithm = ALLREDUCE_NON_OVERLAPPING;
      break;
    case 3:
      *algorithm = ALLREDUCE_RECURSIVE_DOUBLING;
      break;
    case 4:
      *algorithm = ALLREDUCE_RING;
      break;
    case 5:
      *algorithm = ALLREDUCE_RING_SEGMENTED;
      break;
    case 6:
      *algorithm = ALLREDUCE_RABENSEIFNER;
      break;
    case 7:
      *algorithm = ALLREDUCE_ALLGATHER_REDUCE;
      break;
#ifdef OMPI_TEST
    case 8:
      *algorithm = ALLREDUCE_SWING_LAT;
      break;
    case 9:
      *algorithm = ALLREDUCE_SWING_BDW_MEMCPY;
      break;
    case 10:
      *algorithm = ALLREDUCE_SWING_BDW_DT_1;
      break;
    case 11:
      *algorithm = ALLREDUCE_SWING_BDW_DT_2;
      break;
    case 12:
      *algorithm = ALLREDUCE_SWING_BDW_SEG;
      break;
    case 13:
      *algorithm = ALLREDUCE_SWING_BDW_STATIC;
      break;
#endif
    case 14:
      *algorithm = ALLREDUCE_RECURSIVE_DOUBLING_OVER;
      break;
    case 15:
      *algorithm = ALLREDUCE_SWING_LAT_OVER;
      break;
    case 16:
      *algorithm = ALLREDUCE_SWING_BDW_STATIC_OVER;
      break;
    default:
      return -1;
  }
  return 0;
}

static inline int set_allgather_algorithm(allgather_algo_t * algorithm, int algo_number){
  switch (algo_number){
    case 0:
      *algorithm = ALLGATHER_DEFAULT;
      break;
    case 1:
      *algorithm = ALLGATHER_LINEAR;
      break;
    case 2:
      *algorithm = ALLGATHER_K_BRUCK;
      break;
    case 3:
      *algorithm = ALLGATHER_RECURSIVE_DOUBLING;
      break;
    case 4:
      *algorithm = ALLGATHER_RING;
      break;
    case 5:
      *algorithm = ALLGATHER_NEIGHBOR;
      break;
    case 6:
      *algorithm = ALLGATHER_TWO_PROC;
      break;
    case 7:
      *algorithm = ALLGATHER_K_BRUCK_OVER;
      break;
    case 8:
      *algorithm = ALLGATHER_RECURSIVE_DOUBLING_OVER;
      break;
    case 9:
      *algorithm = ALLGATHER_RING_OVER;
      break;
    case 10:
      *algorithm = ALLGATHER_SWING_STATIC_OVER;
      break;
    default:
      return -1;
  }
  return 0;
}

static inline int set_reduce_scatter_algorithm(reduce_scatter_algo_t * algorithm, int algo_number){
  switch (algo_number){
    case 0:
      *algorithm = REDUCE_SCATTER_DEFAULT;
      break;
    case 1:
      *algorithm = REDUCE_SCATTER_NON_OVERLAPPING;
      break;
    case 2:
      *algorithm = REDUCE_SCATTER_RECURSIVE_HALVING;
      break;
    case 3:
      *algorithm = REDUCE_SCATTER_RING;
      break;
    case 4:
      *algorithm = REDUCE_SCATTER_BUTTERFLY;
      break;
    case 5:
      *algorithm = REDUCE_SCATTER_RECURSIVE_HALVING_OVER;
      break;
    case 6:
        *algorithm = REDUCE_SCATTER_RING_OVER;
      break;
    case 7:
      *algorithm = REDUCE_SCATTER_BUTTERFLY_OVER;
      break;
    default:
      return -1;
  }
  return 0;
}

/**
 * @brief Converts a string to a `coll_t` enum value.
 *
 * @param coll_str String representing the collective type (e.g., "ALLREDUCE").
 * @return A `coll_t` enum value corresponding to the input string. Returns `COLL_UNKNOWN` for invalid strings.
 */
static inline coll_t get_collective_from_string(const char *coll_str) {
  if (strcmp(coll_str, "ALLREDUCE") == 0)       return ALLREDUCE;
  if (strcmp(coll_str, "ALLGATHER") == 0)       return ALLGATHER;
  if (strcmp(coll_str, "REDUCE_SCATTER") == 0)  return REDUCE_SCATTER;
  return COLL_UNKNOWN;
}

/**
 * @brief Populates a `routine_decision_t` structure based on environment variables.
 *
 * This function reads the `COLLECTIVE_TYPE` environment variable to determine
 * the collective type. Then, it reads the corresponding `ALGORITHM` environment
 * variable to determine the algorithm for the selected collective.
 *
 * @param test_routine Pointer to a `routine_decision_t` structure to populate.
 * @return `0` on success, `-1` on error (with error messages printed to `stderr`).
 */
int get_routine(routine_decision_t *test_routine, int algorithm) {
  const char *coll_str = NULL;

  // Get the collective type from the environment variable
  coll_str = getenv("COLLECTIVE_TYPE");
  if (NULL == coll_str) {
    fprintf(stderr, "Error! `COLLECTIVE_TYPE` environment variable not set. Aborting...\n");
    return -1;
  }

  // Convert the collective string to a `coll_t` enum value
  test_routine->collective = get_collective_from_string(coll_str);
  if (test_routine->collective == COLL_UNKNOWN) {
    fprintf(stderr, "Error! Invalid `COLLECTIVE_TYPE` value: %s. Aborting...\n", coll_str);
    return -1;
  }

  // Select the algorithm based on the collective type
  switch (test_routine->collective) {
    case ALLREDUCE:
      if (set_allreduce_algorithm(&(test_routine->algorithm.allreduce_algorithm),
                                  algorithm) == -1) {
        fprintf(stderr, "Error! Invalid `ALGORITHM` value for `ALLREDUCE` collective. Aborting...\n");
        return -1;
      }
      break;

    case ALLGATHER:
      if (set_allgather_algorithm(&(test_routine->algorithm.allgather_algorithm),
                                  algorithm) == -1) {
        fprintf(stderr, "Error! Invalid `ALGORITHM` value for `ALLGATHER` collective. Aborting...\n");
        return -1;
      }
      break;

    case REDUCE_SCATTER:
      if (set_reduce_scatter_algorithm(&(test_routine->algorithm.reduce_scatter_algorithm),
                                       algorithm) == -1) {
        fprintf(stderr, "Error! Invalid `ALGORITHM` value for `REDUCE_SCATTER` collective. Aborting...\n");
        return -1;
      }
      break;

    default:
      // This case should never be reached due to prior validation
      fprintf(stderr, "Error! Unknown collective type. Aborting...\n");
      return -1;
  }

  return 0;
}
