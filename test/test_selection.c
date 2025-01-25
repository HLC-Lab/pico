/**
 * @file test_selection.c
 *
 * @brief This file contains the implementation of the functions used
 * to select the routine to test and those to parse the command-line
 * arguments.
 *
 * WARNING:The conditional logic here MUST assure correct selection of
 * the routine to test and check for invalid input combinations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_utils.h"


/**
 * @brief Converts a string to a `coll_t` enum value.
 *
 * @param coll_str String representing the collective type (e.g., "ALLREDUCE").
 * @return A `coll_t` enum value corresponding to the input string.
 *         Returns `COLL_UNKNOWN` for invalid strings.
 */
static inline coll_t get_collective_from_string(const char *coll_str) {
  if (strcmp(coll_str, "ALLREDUCE") == 0)       return ALLREDUCE;
  if (strcmp(coll_str, "ALLGATHER") == 0)       return ALLGATHER;
  if (strcmp(coll_str, "REDUCE_SCATTER") == 0)  return REDUCE_SCATTER;
  return COLL_UNKNOWN;
}


/**
 * @brief Set the allreduce algorithm based on the input integer.
 *
 * @param algorithm Pointer to the allreduce_algo_t element to set.
 * @param coll_number Integer representing the collective type.
 *
 * @return `0` on success, `-1` on error.
 *
 * WARNING: Conditional checks for input validity mus be enforced here.
 */
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


/**
 * @brief Set the allgather algorithm based on the input integer.
 *
 * @param algorithm Pointer to the allgather_algo_t element to set.
 * @param algo_number Integer representing the collective type.
 *
 * @return `0` on success, `-1` on error.
 *
 * WARNING: Conditional checks for input validity mus be enforced here.
 */
static inline int set_allgather_algorithm(allgather_algo_t * algorithm,
                                          int algo_number){
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

/**
 * @brief Set the reduce_scatter algorithm based on the input integer.
 *
 * @param algorithm Pointer to the reduce_scatter_algo_t element to set.
 * @param algo_number Integer representing the collective type.
 *
 * @return `0` on success, `-1` on error.
 *
 * WARNING: Conditional checks for input validity mus be enforced here.
 */
static inline int set_reduce_scatter_algorithm(reduce_scatter_algo_t * algorithm,
                                               int algo_number){
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


int get_routine(routine_decision_t *test_routine, int algorithm) {
  const char *coll_str = NULL;

  // Get the collective type from the environment variable
  coll_str = getenv("COLLECTIVE_TYPE");
  if (NULL == coll_str) {
    fprintf(stderr, "Error! `COLLECTIVE_TYPE` environment \
                    variable not set. Aborting...\n");
    return -1;
  }

  // Convert the collective string to a `coll_t` enum value
  test_routine->collective = get_collective_from_string(coll_str);
  if (test_routine->collective == COLL_UNKNOWN) {
    fprintf(stderr, "Error! Invalid `COLLECTIVE_TYPE` value: \
                     %s. Aborting...\n", coll_str);
    return -1;
  }

  // Select the algorithm based on the collective type
  switch (test_routine->collective) {
    case ALLREDUCE:
      if (set_allreduce_algorithm(&(test_routine->algorithm.allreduce_algorithm),
                                  algorithm) == -1) {
        fprintf(stderr, "Error! Invalid `ALGORITHM` value for \
                        `ALLREDUCE` collective. Aborting...\n");
        return -1;
      }
      break;

    case ALLGATHER:
      if (set_allgather_algorithm(&(test_routine->algorithm.allgather_algorithm),
                                  algorithm) == -1) {
        fprintf(stderr, "Error! Invalid `ALGORITHM` value for \
                        `ALLGATHER` collective. Aborting...\n");
        return -1;
      }
      break;

    case REDUCE_SCATTER:
      if (set_reduce_scatter_algorithm(&(test_routine->algorithm.reduce_scatter_algorithm),
                                       algorithm) == -1) {
        fprintf(stderr, "Error! Invalid `ALGORITHM` value for \
                        `REDUCE_SCATTER` collective. Aborting...\n");
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


int get_command_line_arguments(int argc, char** argv, size_t *array_size, int* iter, const char **type_string, int *alg_number, const char **outputdir) {
  if (argc != 6) {
    fprintf(stderr, "Usage: %s <array_size> <iterations> <dtype> <alg_number> <outputdir>\n", argv[0]);
    return -1;
  }

  char *endptr;
  *array_size = (size_t) strtoll(argv[1], &endptr, 10);
  if (*endptr != '\0' || *array_size <= 0) {
    fprintf(stderr, "Error: Invalid array size. It must be a positive integer. Aborting...\n");
    return -1;
  }

  *iter = (int) strtol(argv[2], &endptr, 10);
  if (*endptr != '\0' || *iter <= 0) {
    fprintf(stderr, "Error: Invalid number of iterations. It must be a positive integer. Aborting...\n");
    return -1;
  }

  *type_string = argv[3];

  *alg_number = (int) strtol(argv[4], &endptr, 10);
  if (*endptr != '\0' || *alg_number < 0) {
    fprintf(stderr, "Error: Invalid alg number. It must be >0. Aborting...\n");
    return -1;
  }

  *outputdir = argv[5];

  return 0;
}


/**
 * @struct TypeMap
 * @brief Maps type names to corresponding MPI data types and sizes.
 */
typedef struct {
  const char* t_string;   /**< Type name as a string. */
  MPI_Datatype mpi_type;  /**< Corresponding MPI datatype. */
  size_t t_size;          /**< Size of the datatype in bytes. */
} TypeMap;

/**
 * @brief Static array mapping string representations to MPI datatypes. Will be
 *        used to map command-line input argument to datatype and its size.
 */
const static TypeMap type_map[] = {
  {"int8",    MPI_INT8_T,   sizeof(int8_t)},
  {"int16",   MPI_INT16_T,  sizeof(int16_t)},
  {"int32",   MPI_INT32_T,  sizeof(int32_t)},
  {"int64",   MPI_INT64_T,  sizeof(int64_t)},
  {"int",     MPI_INT,      sizeof(int)},
  {"float",   MPI_FLOAT,    sizeof(float)},
  {"double",  MPI_DOUBLE,   sizeof(double)},
  {"char",    MPI_CHAR,     sizeof(char)}
};


int get_data_type(const char *type_string, MPI_Datatype *dtype, size_t *type_size) {
  int num_types = sizeof(type_map) / sizeof(type_map[0]);

  for (int i = 0; i < num_types; i++) {
    if (strcmp(type_string, type_map[i].t_string) == 0) {
      *dtype = type_map[i].mpi_type;
      *type_size = type_map[i].t_size;
      return 0;
    }
  }

  fprintf(stderr, "Error: datatype not in type_map. Aborting...\n");
  return -1;
}


