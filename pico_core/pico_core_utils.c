/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <sys/stat.h>
#include <stdbool.h>

#include "pico_core_utils.h"

/**
 * @brief Converts a string to a `coll_t` enum value.
 *
 * @param coll_str String representing the collective type (e.g., "ALLREDUCE").
 * @return A `coll_t` enum value corresponding to the input string.
 *         Returns `COLL_UNKNOWN` for invalid strings.
 */
static inline coll_t get_collective_from_string(const char *coll_str) {
  CHECK_STR(coll_str, "ALLREDUCE", ALLREDUCE);
  CHECK_STR(coll_str, "ALLGATHER", ALLGATHER);
  CHECK_STR(coll_str, "ALLTOALL", ALLTOALL);
  CHECK_STR(coll_str, "BCAST", BCAST);
  CHECK_STR(coll_str, "GATHER", GATHER);
  CHECK_STR(coll_str, "REDUCE", REDUCE);
  CHECK_STR(coll_str, "REDUCE_SCATTER", REDUCE_SCATTER);
  CHECK_STR(coll_str, "SCATTER", SCATTER);
  return COLL_UNKNOWN;
}

/**
* @brief Select and returns the appropriate allocator function based
* on the collective type. It returns NULL if the collective type is 
* not supported.
*
* @param collectove `coll_t` enum value representing the collective type.
*
* @return Pointer to the selected allocator function, or NULL if the
*         collective type is not supported.
*/
static inline allocator_func_ptr get_allocator(coll_t collective) {
  switch (collective) {
    case ALLREDUCE:
      return allreduce_allocator;
    case ALLGATHER:
      return allgather_allocator;
    case ALLTOALL:
      return alltoall_allocator;
    case BCAST:
      return bcast_allocator;
    case GATHER:
      return gather_allocator;
    case REDUCE:
      return reduce_allocator;
    case REDUCE_SCATTER:
      return reduce_scatter_allocator;
    case SCATTER:
      return scatter_allocator;
    default:
      return NULL;
  }
}

#ifdef CUDA_AWARE
static inline allocator_func_ptr get_allocator_cuda(coll_t collective) {
  switch (collective) {
    case ALLREDUCE:
      return allreduce_allocator_cuda;
    case ALLGATHER:
      return allgather_allocator_cuda;
    case ALLTOALL:
      return alltoall_allocator_cuda;
    case BCAST:
      return bcast_allocator_cuda;
    case GATHER:
      return gather_allocator_cuda;
    case REDUCE:
      return reduce_allocator_cuda;
    case REDUCE_SCATTER:
      return reduce_scatter_allocator_cuda;
    case SCATTER:
      return scatter_allocator_cuda;
    default:
      return NULL;
  }
}
#endif

/**
* @brief Select and returns the appropriate allreduce function based
* on the algorithm. It returns the default allreduce function if the
* algorithm is internal.
*
* WARNING: It does not check if the algorithm is supported and always
* defauls to the internal allreduce function.
*/
static inline allreduce_func_ptr get_allreduce_function(const char *algorithm) {
  CHECK_STR(algorithm, "recursive_doubling_over", allreduce_recursivedoubling);
  CHECK_STR(algorithm, "ring_over", allreduce_ring);
  CHECK_STR(algorithm, "rabenseifner_over", allreduce_rabenseifner);
  CHECK_STR(algorithm, "bine_lat_over", allreduce_bine_lat);
  CHECK_STR(algorithm, "bine_bdw_static_over", allreduce_bine_bdw_static);
  CHECK_STR(algorithm, "bine_bdw_remap_over", allreduce_bine_bdw_remap);
  CHECK_STR(algorithm, "bine_bdw_remap_segmented_over", allreduce_bine_bdw_remap_segmented);
  CHECK_STR(algorithm, "bine_block_by_block_any_even", allreduce_bine_block_by_block_any_even);

  PICO_CORE_DEBUG_PRINT_STR("MPI_Allreduce");
  return allreduce_wrapper;
}

/**
* @brief Select and returns the appropriate allgather function based
* on the algorithm. It returns the default allgather function if the
* algorithm is internal.
*
* WARNING: It does not check if the algorithm is supported and always
* defauls to the internal allgather function.
*/
static inline allgather_func_ptr get_allgather_function(const char *algorithm) {
  CHECK_STR(algorithm, "k_bruck_over", allgather_k_bruck);
  CHECK_STR(algorithm, "recursive_doubling_over", allgather_recursivedoubling);
  CHECK_STR(algorithm, "ring_over", allgather_ring);
  CHECK_STR(algorithm, "sparbit_over", allgather_sparbit);
  CHECK_STR(algorithm, "bine_block_by_block_over_any_even", allgather_bine_block_by_block_any_even);
  CHECK_STR(algorithm, "bine_block_by_block_over", allgather_bine_block_by_block);
  CHECK_STR(algorithm, "bine_permute_static_over", allgather_bine_permute_static);
  CHECK_STR(algorithm, "bine_send_static_over", allgather_bine_send_static);
  CHECK_STR(algorithm, "bine_permute_remap_over", allgather_bine_permute_remap);
  CHECK_STR(algorithm, "bine_send_remap_over", allgather_bine_send_remap);
  CHECK_STR(algorithm, "bine_2_blocks_over", allgather_bine_2_blocks);
  CHECK_STR(algorithm, "bine_2_blocks_dtype_over", allgather_bine_2_blocks_dtype);

  PICO_CORE_DEBUG_PRINT_STR("MPI_Allgather");
  return allgather_wrapper;
}

/**
* @brief Select and returns the appropriate alltoall function based
* on the algorithm. It returns the default alltoall function if the
* algorithm is internal.
*
* WARNING: It does not check if the algorithm is supported and always
* defauls to the internal alltoall function.
*/
static inline alltoall_func_ptr get_alltoall_function(const char *algorithm) {
  CHECK_STR(algorithm, "bine_over", alltoall_bine);

  PICO_CORE_DEBUG_PRINT_STR("MPI_Alltoall");
  return alltoall_wrapper;
}

/**
* @brief Select and returns the appropriate bcast function based
* on the algorithm. It returns the default bcast function if the
* algorithm is internal.
*
* WARNING: It does not check if the algorithm is supported and always
* defauls to the internal bcast function.
*/
static inline bcast_func_ptr get_bcast_function(const char *algorithm) {
  CHECK_STR(algorithm, "scatter_allgather_over", bcast_scatter_allgather);
  CHECK_STR(algorithm, "bine_lat_over", bcast_bine_lat);
  CHECK_STR(algorithm, "bine_lat_reversed_over", bcast_bine_lat_reversed);
  CHECK_STR(algorithm, "bine_lat_new_over", bcast_bine_lat_new);
  CHECK_STR(algorithm, "bine_lat_i_new_over", bcast_bine_lat_i_new);
  CHECK_STR(algorithm, "bine_bdw_static_over", bcast_bine_bdw_static);
  // CHECK_STR(algorithm, "bine_bdw_static_reversed_over", bcast_bine_bdw_static_reversed);
  CHECK_STR(algorithm, "bine_bdw_remap_over", bcast_bine_bdw_remap);

  PICO_CORE_DEBUG_PRINT_STR("MPI_Bcast");
  return bcast_wrapper;
}


/**
* @brief Select and returns the appropriate gather function based
* on the algorithm. It returns the default gather function if the
* algorithm is internal.
*
* WARNING: It does not check if the algorithm is supported and always
* defauls to the internal gather function.
*/
static inline gather_func_ptr get_gather_function(const char *algorithm) {
  CHECK_STR(algorithm, "bine_over", gather_bine);

  PICO_CORE_DEBUG_PRINT_STR("MPI_Gather");
  return gather_wrapper;
}


/**
* @brief Select and returns the appropriate reduce function based
* on the algorithm. It returns the default reduce function if the
* algorithm is internal.
*
* WARNING: It does not check if the algorithm is supported and always
* defauls to the internal reduce function.
*/
static inline reduce_func_ptr get_reduce_function(const char *algorithm) {
  CHECK_STR(algorithm, "bine_lat_over", reduce_bine_lat);
  CHECK_STR(algorithm, "bine_bdw_over", reduce_bine_bdw);

  PICO_CORE_DEBUG_PRINT_STR("MPI_Reduce");
  return reduce_wrapper;
}

/**
* @breif Select and returns the appropriate reduce scatter function based
* on the algorithm. It returns the default reduce scatter function if the
* algorithm is internal.
*
* WARNING: It does not check if the algorithm is supported and always
* defauls to the internal reduce scatter function.
*/
static inline reduce_scatter_func_ptr get_reduce_scatter_function (const char *algorithm){
  CHECK_STR(algorithm, "recursive_halving_over", reduce_scatter_recursivehalving);
  CHECK_STR(algorithm, "recursive_distance_doubling_over", reduce_scatter_recursive_distance_doubling);
  CHECK_STR(algorithm, "ring_over", reduce_scatter_ring);
  CHECK_STR(algorithm, "butterfly_over", reduce_scatter_butterfly);
  CHECK_STR(algorithm, "bine_static_over", reduce_scatter_bine_static);
  CHECK_STR(algorithm, "bine_send_remap_over", reduce_scatter_bine_send_remap);
  CHECK_STR(algorithm, "bine_permute_remap_over", reduce_scatter_bine_permute_remap);  
  CHECK_STR(algorithm, "bine_block_by_block_over", reduce_scatter_bine_block_by_block);
  CHECK_STR(algorithm, "bine_block_by_block_any_even", reduce_scatter_bine_block_by_block_any_even);

  PICO_CORE_DEBUG_PRINT_STR("MPI_Reduce_scatter");
  return MPI_Reduce_scatter;
}

/**
* @breif Select and returns the appropriate scatter function based
* on the algorithm. It returns the default scatter function if the
* algorithm is internal.
*
* WARNING: It does not check if the algorithm is supported and always
* defauls to the internal scatter function.
*/
static inline scatter_func_ptr get_scatter_function (const char *algorithm){
  CHECK_STR(algorithm, "bine_over", scatter_bine);

  PICO_CORE_DEBUG_PRINT_STR("MPI_Scatter");
  return scatter_wrapper;
}

int get_routine(test_routine_t *test_routine, const char *algorithm) {
  const char *coll_str = NULL, *is_segmented, *segsize = NULL;

  // Get the collective type from the environment variable
  coll_str = getenv("COLLECTIVE_TYPE");
  if(NULL == coll_str) {
    fprintf(stderr, "Error! `COLLECTIVE_TYPE` environment \
                    variable not set. Aborting...");
    return -1;
  }

  // Convert the collective string to a `coll_t` enum value
  test_routine->collective = get_collective_from_string(coll_str);
  if(test_routine->collective == COLL_UNKNOWN) {
    fprintf(stderr, "Error! Invalid `COLLECTIVE_TYPE` value: \
                     %s. Aborting...", coll_str);
    return -1;
  }

  // Set the right allocator based on the collective type
  test_routine->allocator = get_allocator(test_routine->collective);
  if(NULL == test_routine->allocator) {
    fprintf(stderr, "Error! Allocator is NULL. Aborting...");
    return -1;
  }
  #ifdef CUDA_AWARE
  test_routine->allocator_cuda = get_allocator_cuda(test_routine->collective);
  #endif

  // Set the right function pointer based on the collective type and algorithm
  switch (test_routine->collective){
    case ALLREDUCE:
      test_routine->function.allreduce = get_allreduce_function(algorithm);
      break;
    case ALLGATHER:
      test_routine->function.allgather = get_allgather_function(algorithm);
      break;
    case ALLTOALL:
      test_routine->function.alltoall = get_alltoall_function(algorithm);
      break;
    case BCAST:
      test_routine->function.bcast = get_bcast_function(algorithm);
      break;
    case GATHER:
      test_routine->function.gather = get_gather_function(algorithm);
      break;
    case REDUCE:
      test_routine->function.reduce = get_reduce_function(algorithm);
      break;
    case REDUCE_SCATTER:
      test_routine->function.reduce_scatter = get_reduce_scatter_function(algorithm);
      break;
    case SCATTER:
      test_routine->function.scatter = get_scatter_function(algorithm);
      break;
    default :
      fprintf(stderr, "Error! Invalid collective type. Aborting...");
      return -1;
  }

  is_segmented = getenv("SEGMENTED");
  if(strcmp(is_segmented, "yes") == 0) { 
    segsize = getenv("SEGSIZE");
    if(segsize == NULL) {
      return -1;
    }
    bine_allreduce_segsize = (size_t) strtoll(segsize, NULL, 10);
  }

  test_routine->segsize = bine_allreduce_segsize;

  return 0;
}


  // if(rank == 0){
  //   char data_filename[128], data_fullpath[PICO_CORE_MAX_PATH_LENGTH];
  //   if(bine_allreduce_segsize != 0){
  //     snprintf(data_filename, sizeof(data_filename), "/%ld_%s_%ld_%s.csv",
  //              count, algorithm, bine_allreduce_segsize, type_string);
  //   } else {
  //     snprintf(data_filename, sizeof(data_filename), "/%ld_%s_%s.csv",
  //              count, algorithm, type_string);
  //   }
  //   if(concatenate_path(data_dir, data_filename, data_fullpath) == -1) {
  //     line = __LINE__;
  //     goto err_hndl;
  //   }
  //   if(write_output_to_file(output_level, data_fullpath, highest, all_times, iter) == -1){
  //     line = __LINE__;
  //     goto err_hndl;
  //   }
  // }
  //
  // // Save to file allocations (it uses MPI parallel I/O operations)
  // char alloc_filename[128] = "alloc.csv";
  // char alloc_fullpath[PICO_CORE_MAX_PATH_LENGTH];
  // if(concatenate_path(outputdir, alloc_filename, alloc_fullpath) == -1){
  //   line = __LINE__;
  //   goto err_hndl;
  // }
int get_data_saving_options(test_routine_t *test_routine, size_t count,
                            const char *algorithm, const char *type_string) {
  int comm_sz;
  const char *output_level = NULL, *output_dir = NULL, *data_dir = NULL;
  char data_filename[128], alloc_filename[128];

  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  output_dir = getenv("OUTPUT_DIR");
  data_dir = getenv("DATA_DIR");
  output_level = getenv("OUTPUT_LEVEL");
  if(output_dir == NULL || data_dir == NULL || output_level == NULL) {
    fprintf(stderr, "Error: Environment variables OUTPUT_DIR, DATA_DIR or OUTPUT_LEVEL not set. Aborting...");
    return -1;
  }

  if (test_routine->segsize != 0) {
    snprintf(data_filename, sizeof(data_filename), "/%ld_%s_%ld_%s.csv", count, algorithm, test_routine->segsize, type_string);
  } else {
    snprintf(data_filename, sizeof(data_filename), "/%ld_%s_%s.csv", count, algorithm, type_string);
  }

  if(concatenate_path(data_dir, data_filename, test_routine->output_data_file) == -1) {
    fprintf(stderr, "Error: Failed to concatenate path. Aborting...");
    return -1;
  }

#ifndef CUDA_AWARE
  snprintf(alloc_filename, sizeof(alloc_filename), "/alloc_%d.csv", comm_sz);
#else
  snprintf(alloc_filename, sizeof(alloc_filename), "/alloc_%d_GPU.csv", comm_sz);
#endif

  if (concatenate_path(output_dir, alloc_filename, test_routine->alloc_file) == -1) {
    fprintf(stderr, "Error: Failed to concatenate path. Aborting...");
    return -1;
  }

  if(strcmp(output_level, "all") == 0) {
    test_routine->output_level = ALL;
  } else if(strcmp(output_level, "statistics") == 0) {
    test_routine->output_level = STATISTICS;
  } else if(strcmp(output_level, "summarized") == 0) {
    test_routine->output_level = SUMMARIZED;
  } else {
    fprintf(stderr, "Error: Invalid OUTPUT_LEVEL value. Aborting...");
    return -1;
  }

  return 0;
}


#ifdef CUDA_AWARE
int coll_memcpy_host_to_device(void** d_buf, void** buf, size_t count, size_t type_size, coll_t coll) {

  int comm_sz, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cudaError_t err;

  switch (coll) {
    case ALLREDUCE:
      PICO_CORE_CUDA_CHECK(cudaMemcpy(*d_buf, *buf, count * type_size, cudaMemcpyHostToDevice), err);
      break;
    case ALLGATHER:
      PICO_CORE_CUDA_CHECK(cudaMemcpy(*d_buf, *buf, (count / (size_t) comm_sz) * type_size, cudaMemcpyHostToDevice), err);
      break;
    case ALLTOALL:
      PICO_CORE_CUDA_CHECK(cudaMemcpy(*d_buf, *buf, count * type_size, cudaMemcpyHostToDevice), err);
      break;
    case BCAST:
      if (rank == 0) {
        PICO_CORE_CUDA_CHECK(cudaMemcpy(*d_buf, *buf, count * type_size, cudaMemcpyHostToDevice), err);
      }
      break;
    case GATHER:
      PICO_CORE_CUDA_CHECK(cudaMemcpy(*d_buf, *buf, (count / (size_t) comm_sz) * type_size, cudaMemcpyHostToDevice), err);
      break;
    case REDUCE:
      PICO_CORE_CUDA_CHECK(cudaMemcpy(*d_buf, *buf, count * type_size, cudaMemcpyHostToDevice), err);
      break;
    case REDUCE_SCATTER;
      PICO_CORE_CUDA_CHECK(cudaMemcpy(*d_buf, *buf, count * type_size, cudaMemcpyHostToDevice), err);
      break;
    case SCATTER:
      if (rank == 0) {
        PICO_CORE_CUDA_CHECK(cudaMemcpy(*d_buf, *buf, count * type_size, cudaMemcpyHostToDevice), err);
      }
      break;
    default:
      fprintf(stderr, "Error: Unsupported collective type. Aborting...");
      return -1;
  }

  return 0;
}

int coll_memcpy_device_to_host(void** d_buf, void** buf, size_t count, size_t type_size, coll_t coll) {

  int comm_sz, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cudaError_t err;

  switch (coll) {
    case ALLREDUCE:
      PICO_CORE_CUDA_CHECK(cudaMemcpy(*buf, *d_buf, count * type_size, cudaMemcpyDeviceToHost), err);
      break;
    case ALLGATHER:
      PICO_CORE_CUDA_CHECK(cudaMemcpy(*buf, *d_buf, count * type_size, cudaMemcpyDeviceToHost), err);
      break;
    case ALLTOALL:
      PICO_CORE_CUDA_CHECK(cudaMemcpy(*buf, *d_buf, count * type_size, cudaMemcpyDeviceToHost), err);
      break;
    case BCAST:
      if (rank != 0) {
        PICO_CORE_CUDA_CHECK(cudaMemcpy(*buf, *d_buf, count * type_size, cudaMemcpyDeviceToHost), err);
      }
      break;
    case GATHER:
      if (rank == 0) {
        PICO_CORE_CUDA_CHECK(cudaMemcpy(*buf, *d_buf, count * type_size, cudaMemcpyDeviceToHost), err);
      }
      break;
    case REDUCE:
      if (rank == 0) {
        PICO_CORE_CUDA_CHECK(cudaMemcpy(*buf, *d_buf, count * type_size, cudaMemcpyDeviceToHost), err);
      }
      break;
    case REDUCE_SCATTER;
      PICO_CORE_CUDA_CHECK(cudaMemcpy(*buf, *d_buf, (count / (size_t) comm_sz) * type_size, cudaMemcpyDeviceToHost), err);
      break;
    case SCATTER:
      PICO_CORE_CUDA_CHECK(cudaMemcpy(*buf, *d_buf, (count / (size_t) comm_sz) * type_size, cudaMemcpyDeviceToHost), err);
      break;
    default:
      fprintf(stderr, "Error: Unsupported collective type. Aborting...");
      return -1;
  }

  return 0;
}
#endif // CUDA_AWARE

int test_loop(test_routine_t test_routine, void *sbuf, void *rbuf, size_t count,
              MPI_Datatype dtype, MPI_Comm comm, int iter, double *times){
  int rank, comm_sz, ret, *rcounts = NULL;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);

  size_t local_count = count / (size_t) comm_sz; /**< Rank count for collectives with distributed data */

  switch (test_routine.collective){
    case ALLREDUCE:
      ret = allreduce_test_loop(sbuf, rbuf, count, dtype, MPI_SUM,
                            comm, iter, times, test_routine);
      break;
    case ALLGATHER:
      ret = allgather_test_loop(sbuf, local_count, dtype,
                            rbuf, local_count, dtype,
                            comm, iter, times, test_routine);
      break;
    case ALLTOALL:
      ret = alltoall_test_loop(sbuf, local_count, dtype,
                           rbuf, local_count, dtype,
                           comm, iter, times, test_routine);
    break;
    case BCAST:
      ret = bcast_test_loop(sbuf, count, dtype, 0, comm, iter, times,
                        test_routine);
      break;
    case GATHER:
      ret = gather_test_loop(sbuf, local_count, dtype,
                         rbuf, local_count, dtype, 0, comm,
                         iter, times, test_routine);
      break;
    case REDUCE:
      ret = reduce_test_loop(sbuf, rbuf, count, dtype, MPI_SUM, 0, comm,
                             iter, times, test_routine);
      break;
    case REDUCE_SCATTER:
      rcounts = (int *)malloc(comm_sz * sizeof(int));
      for(int i = 0; i < comm_sz; i++) { rcounts[i] = local_count; }
      ret = reduce_scatter_test_loop(sbuf, rbuf, rcounts, dtype, MPI_SUM, comm,
                                iter, times, test_routine);
      free(rcounts);
      break;
    case SCATTER:
      ret = scatter_test_loop(sbuf, local_count, dtype,
                          rbuf, local_count, dtype,
                          0, comm, iter, times, test_routine);
      break;
    default:
      fprintf(stderr, "still not implemented, aborting...");
      return -1;
  }
  return ret;
}

int ground_truth_check(test_routine_t test_routine, void *sbuf, void *rbuf,
                       void *rbuf_gt, size_t count, MPI_Datatype dtype, MPI_Comm comm){
  int rank, comm_sz, *rcounts = NULL, ret = 0, type_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  MPI_Type_size(dtype, &type_size);

  switch (test_routine.collective){
    case ALLREDUCE:
      PMPI_Allreduce(sbuf, rbuf_gt, count, dtype, MPI_SUM, comm);
      GT_CHECK_BUFFER(rbuf, rbuf_gt, count, dtype, comm);
      break;
    case ALLGATHER:
      PMPI_Allgather(sbuf, count / (size_t) comm_sz, dtype, \
                 rbuf_gt, count / (size_t) comm_sz, dtype, comm);
      GT_CHECK_BUFFER(rbuf, rbuf_gt, count, dtype, comm);
      break;
    case ALLTOALL:
      PMPI_Alltoall(sbuf, count / (size_t) comm_sz, dtype, \
             rbuf_gt, count / (size_t) comm_sz, dtype, comm);
      GT_CHECK_BUFFER(rbuf, rbuf_gt, count / (size_t) comm_sz, dtype, comm);
      break;
    case BCAST:
      if(rank == 0) {
        memcpy(rbuf_gt, sbuf, count * type_size);
      }
      PMPI_Bcast(rbuf_gt, count, dtype, 0, comm);
      GT_CHECK_BUFFER(sbuf, rbuf_gt, count, dtype, comm);
      break;
    case GATHER:
      PMPI_Gather(sbuf, count / (size_t) comm_sz, dtype, rbuf_gt, count / (size_t) comm_sz, dtype, 0, comm);
      if (rank == 0) {
        GT_CHECK_BUFFER(rbuf, rbuf_gt, count, dtype, comm);
      }
      break;
    case REDUCE:
      PMPI_Reduce(sbuf, rbuf_gt, count, dtype, MPI_SUM, 0, comm);
      if(rank == 0){ 
        GT_CHECK_BUFFER(rbuf, rbuf_gt, count, dtype, comm);
      }
      break;
    case REDUCE_SCATTER:
      rcounts = (int *)malloc(comm_sz * sizeof(int));
      for(int i = 0; i < comm_sz; i++) { rcounts[i] = count / comm_sz; }
      PMPI_Reduce_scatter(sbuf, rbuf_gt, rcounts, dtype, MPI_SUM, comm);
      GT_CHECK_BUFFER(rbuf, rbuf_gt, rcounts[rank], dtype, comm);
      free(rcounts);
      break;
    case SCATTER:
      PMPI_Scatter(sbuf, count / (size_t) comm_sz, dtype, rbuf_gt, count / (size_t) comm_sz, dtype, 0, comm);
      GT_CHECK_BUFFER(rbuf, rbuf_gt, count / (size_t) comm_sz, dtype, comm);
      break;
    default:
      fprintf(stderr, "still not implemented, aborting...");
      return -1;
  }
  return ret;
}


int get_command_line_arguments(int argc, char** argv, size_t *array_count, int* iter,
                               const char **algorithm, const char **type_string) {
  if(argc != 5) {
    fprintf(stderr, "Usage: %s <array_count> <iterations> <algorithm> <dtype>", argv[0]);
    return -1;
  }

  char *endptr;
  *array_count = (size_t) strtoll(argv[1], &endptr, 10);
  if(*endptr != '\0' || *array_count <= 0) {
    fprintf(stderr, "Error: Invalid array count. It must be a positive integer. Aborting...");
    return -1;
  }

  *iter = (int) strtol(argv[2], &endptr, 10);
  if(*endptr != '\0' || *iter <= 0) {
    fprintf(stderr, "Error: Invalid number of iterations. It must be a positive integer. Aborting...");
    return -1;
  }

  *algorithm = argv[3];

  *type_string = argv[4];

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
  {"int8",          MPI_INT8_T,         sizeof(int8_t)},
  {"int16",         MPI_INT16_T,        sizeof(int16_t)},
  {"int32",         MPI_INT32_T,        sizeof(int32_t)},
  {"int64",         MPI_INT64_T,        sizeof(int64_t)},
  {"int",           MPI_INT,            sizeof(int)},
  {"float",         MPI_FLOAT,          sizeof(float)},
  {"double",        MPI_DOUBLE,         sizeof(double)},
  {"char",          MPI_CHAR,           sizeof(char)},
  {"unsigned_char", MPI_UNSIGNED_CHAR,  sizeof(unsigned char)}
};


int get_data_type(const char *type_string, MPI_Datatype *dtype, size_t *type_size) {
  int num_types = sizeof(type_map) / sizeof(type_map[0]);

  for(int i = 0; i < num_types; i++) {
    if(strcmp(type_string, type_map[i].t_string) == 0) {
      *dtype = type_map[i].mpi_type;
      *type_size = type_map[i].t_size;
      return 0;
    }
  }

  fprintf(stderr, "Error: datatype %s not in `type_map`. Aborting...", type_string);
  return -1;
}


int split_communicator(MPI_Comm *inter_comm, MPI_Comm *intra_comm){
  int rank, size, current_tasks_per_node;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  char* tasks_per_node_env = getenv("CURRENT_TASKS_PER_NODE");
  if (tasks_per_node_env == NULL) {
    fprintf(stderr, "Error: CURRENT_TASKS_PER_NODE environment variable is not set.\n");
    return MPI_ERR_COMM;
  }
  current_tasks_per_node = atoi(tasks_per_node_env);
  if (current_tasks_per_node <= 0) {
    fprintf(stderr, "Error: CURRENT_TASKS_PER_NODE must be a positive integer.\n");
    return MPI_ERR_COMM;
  }

  if (size % current_tasks_per_node != 0) {
    fprintf(stderr, "Error: The number of ranks (%d) is not divisible by CURRENT_TASKS_PER_NODE (%d).\n", size, current_tasks_per_node);
    return MPI_ERR_COMM;
  }

  MPI_Comm_split(MPI_COMM_WORLD, (rank / current_tasks_per_node), rank, intra_comm);
  MPI_Comm_split(MPI_COMM_WORLD, (rank % current_tasks_per_node), rank, inter_comm);

  return MPI_SUCCESS;
}

/**
* @breif Writes all timing results to a specified output file in CSV format.
*
* @param fullpath  The full path to the output file.
* @param highest   An array containing the highest timing values for each iteration.
* @param all_times A 2D array flattened into 1D containing timing values for all ranks
*                  across all iterations.
* @param iter      The number of iterations.
*
* @return int Returns 0 on success, or -1 if an error occurs.
* @note Time is saved in ns (i.e. 10^-9 s).
*/
static inline int write_all_output_to_file(const char *fullpath, double *highest, double *all_times, int iter) {
  int comm_sz;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  FILE *output_file = fopen(fullpath, "w");
  if(output_file == NULL) {
    fprintf(stderr, "Error: Opening file %s for writing", fullpath);
    return -1;
  }

  // Write the header with ranks from rank0 to rankN
  fprintf(output_file, "highest");
  for(int rank = 0; rank < comm_sz; rank++) {
    fprintf(output_file, ",rank%d", rank);
  }
  fprintf(output_file, "\n");

  // Write the timing data
  for(int i = 0; i < iter; i++) {
    fprintf(output_file, "%" PRId64, (int64_t)(highest[i] * 1e9));
    for(int j = 0; j < comm_sz; j++) {
      fprintf(output_file, ",%" PRId64, (int64_t)(all_times[j * iter + i] * 1e9));
    }
    fprintf(output_file, "\n");
  }

  fclose(output_file);
  return 0;
}

/**
* @brief Writes the summarized timing results to a specified output file in CSV format.
*
* @param fullpath The full path to the output file.
* @param highest An array containing the highest timing values for each iteration.
* @param iter The number of iterations.
*
* @return int Returns 0 on success, or -1 if an error occurs.
*
* @note Time is saved in ns (i.e. 10^-9 s).
*/
static inline int write_summarized_output_to_file(const char *fullpath, double *highest, int iter){
  FILE *output_file = fopen(fullpath, "w");
  if(output_file == NULL) {
    fprintf(stderr, "Error: Opening file %s for writing", fullpath);
    return -1;
  }

  // Write the header with ranks from rank0 to rankN
  fprintf(output_file, "highest\n");

  // Write the timing data
  for(int i = 0; i < iter; i++) {
    fprintf(output_file, "%" PRId64"\n", (int64_t)(highest[i] * 1e9));
  }

  fclose(output_file);
  return 0;
}

int write_statistics_output_to_file(const char *fullpath, double *highest, int iter) {
  // TODO: Implement the statistics output file writing
  return 0;
}

int write_output_to_file(test_routine_t test_routine, double *highest, double *all_times, int iter)
{
  switch (test_routine.output_level) {
    case ALL:
      return write_all_output_to_file(test_routine.output_data_file, highest, all_times, iter);
    case STATISTICS:
      return write_statistics_output_to_file(test_routine.output_data_file, highest, iter);
    case SUMMARIZED:
      return write_summarized_output_to_file(test_routine.output_data_file, highest, iter);
    default:
      fprintf(stderr, "Error: Output level not recognized. Aborting...");
      return -1;
  }
}

int file_not_exists(const char* filename) {
  struct stat buffer;
  return (stat(filename, &buffer) != 0) ? 1 : 0;
}

static inline void trim_newline(char *str) {
    str[strcspn(str, "\r\n")] = '\0';
}


int write_allocations_to_file(const char* filename, MPI_Comm comm) {
  int rank, comm_sz, name_len;
  const char *location = NULL;
  char processor_name[MPI_MAX_PROCESSOR_NAME], local_entry[PICO_CORE_MAX_ALLOC_NAME_LEN];
  char* gather_buffer = NULL;
  
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  MPI_Get_processor_name(processor_name, &name_len);

  location = getenv("LOCATION");
  if (location == NULL) {
      fprintf(stderr, "Error: `LOCATION` environment variable not set. Aborting...\n");
      return -1;
  }

  memset(local_entry, 0, sizeof(local_entry));
  if (strcmp(location, "lumi") == 0) {
    FILE *xname_file = fopen("/etc/cray/xname", "r");
    char xname[32];
    if (xname_file == NULL) {
      fprintf(stderr, "Error: Could not open /etc/cray/xname for reading.\n");
      return -1;
    }
    fgets(xname, sizeof(xname), xname_file);
    fclose(xname_file);
    trim_newline(xname);

    snprintf(local_entry, sizeof(local_entry), "%d,%s,%s\n", rank, processor_name, xname);
  } else {
    snprintf(local_entry, sizeof(local_entry), "%d,%s\n", rank, processor_name);
  }

  if (rank == 0) {
    gather_buffer = malloc(comm_sz * PICO_CORE_MAX_ALLOC_NAME_LEN * sizeof(char));
    if (gather_buffer == NULL) {
      fprintf(stderr, "Error: Unable to allocate gather_buffer.\n");
      return -1;
    }
  }

  MPI_Gather(local_entry, PICO_CORE_MAX_ALLOC_NAME_LEN, MPI_CHAR,
              gather_buffer, PICO_CORE_MAX_ALLOC_NAME_LEN, MPI_CHAR,
              0, comm);

  if (rank == 0) {
    FILE *output_file = fopen(filename, "w");
    if (output_file == NULL) {
      fprintf(stderr, "Error: Could not open file %s for writing.\n", filename);
      free(gather_buffer);
      return -1;
    }
    if (strcmp(location, "lumi") == 0) {
      fprintf(output_file, "%s", PICO_CORE_HEADER_LUMI);
    } else {
      fprintf(output_file, "%s", PICO_CORE_HEADER_DEFAULT);
    }

    for (int i = 0; i < comm_sz; i++) {
      char *entry = gather_buffer + i * PICO_CORE_MAX_ALLOC_NAME_LEN;
      int actual_length = strlen(entry);
      fprintf(output_file, "%.*s", actual_length, entry);
    }
    fclose(output_file);
    free(gather_buffer);
  }

  return MPI_SUCCESS;
}


int rand_sbuf_generator(void *sbuf, MPI_Datatype dtype, size_t count,
                         MPI_Comm comm, test_routine_t test_routine) {
  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  unsigned int seed = time(NULL) + rank;
  
  // For BCAST and SCATTER, only rank 0 generates the sendbuf
  if((test_routine.collective == BCAST || test_routine.collective == SCATTER)
      && rank != 0) {
    return 0;
  }

  size_t real_sbuf_count =
    (test_routine.collective == ALLGATHER ||
     test_routine.collective == GATHER    ||
     test_routine.collective == ALLTOALL) ?
                                        count / (size_t) comm_sz : count;

  for(size_t i = 0; i < real_sbuf_count; i++) {
    if(dtype == MPI_INT8_T) {
      ((int8_t *)sbuf)[i] = (int8_t)((rand_r(&seed) % 256) - 128);
    } else if(dtype == MPI_INT16_T) {
      ((int16_t *)sbuf)[i] = (int16_t)((rand_r(&seed) % 65536) - 32768);
    } else if(dtype == MPI_INT32_T) {
      ((int32_t *)sbuf)[i] = (int32_t)(rand_r(&seed));
    } else if(dtype == MPI_INT64_T) {
      ((int64_t *)sbuf)[i] = ((int64_t)rand_r(&seed) << 32) | rand_r(&seed);
    } else if(dtype == MPI_INT) {
      ((int *)sbuf)[i] = (int)rand_r(&seed);
    } else if(dtype == MPI_FLOAT) {
      ((float *)sbuf)[i] = (float)rand_r(&seed) / (float) RAND_MAX * 100.0f;
    } else if(dtype == MPI_DOUBLE) {
      ((double *)sbuf)[i] = (double)rand_r(&seed) / (double) RAND_MAX * 100.0;
    } else if(dtype == MPI_CHAR) {
      ((char *)sbuf)[i] = (char)((rand_r(&seed) % 256) - 128);
    } else if(dtype == MPI_UNSIGNED_CHAR) {
      ((unsigned char *)sbuf)[i] = (unsigned char)(rand_r(&seed) % 256);
    } else {
      fprintf(stderr, "Error: sbuf not generated correctly. Aborting...");
      return -1;
    }
  }

  return 0;
}


int concatenate_path(const char *dir_path, const char *filename, char *fullpath) {
  if(dir_path == NULL || filename == NULL) {
    fprintf(stderr, "Directory path or filename is NULL.");
    return -1;
  }

  size_t dir_path_len = strlen(dir_path);
  size_t filename_len = strlen(filename);

  if(dir_path_len == 0) {
    fprintf(stderr, "Directory path is empty.");
    return -1;
  }

  if(dir_path_len + filename_len + 2 > PICO_CORE_MAX_PATH_LENGTH) {
    fprintf(stderr, "Combined path length exceeds buffer size.");
    return -1;
  }

  strcpy(fullpath, dir_path);
  if(dir_path[dir_path_len - 1] != '/') {
    strcat(fullpath, "/");
  }
  strcat(fullpath, filename);

  return 0;
}


int are_equal_eps(const void *buf_1, const void *buf_2, size_t count,
                  MPI_Datatype dtype, MPI_Comm comm) {
  int comm_sz;
  MPI_Comm_size(comm, &comm_sz);

  if(count == 0) return 0;

  if(MPI_FLOAT == dtype) {
    float *b1 = (float *) buf_1;
    float *b2 = (float *) buf_2;

    float epsilon = comm_sz * PICO_CORE_BASE_EPSILON_FLOAT * 100.0f;

    for(size_t i = 0; i < count; i++) {
      if(fabs(b1[i] - b2[i]) > epsilon) {
        return -1;
      }
    }
  } else if(MPI_DOUBLE == dtype) {
    double *b1 = (double *) buf_1;
    double *b2 = (double *) buf_2;

    double epsilon = comm_sz * PICO_CORE_BASE_EPSILON_DOUBLE * 100.0;

    for(size_t i = 0; i < count; i++) {
      if(fabs(b1[i] - b2[i]) > epsilon) {
        return -1;
      }
    }
  }

  return 0;
}

/**
 * Helper function to print an array of elements based on MPI_Datatype.
 */
static inline void print_buffer_helper(const void *buf, size_t count, MPI_Datatype dtype) {
  if(dtype == MPI_INT64_T) {
    const int64_t *data = (const int64_t *)buf;
    for(size_t j = 0; j < count; j++) {
      printf("%" PRId64 " ", data[j]);
    }
  } else if(dtype == MPI_INT32_T) {
    const int32_t *data = (const int32_t *)buf;
    for(size_t j = 0; j < count; j++) {
      printf("%" PRId32 " ", data[j]);
    }
  } else if(dtype == MPI_INT) {
    const int *data = (const int *)buf;
    for(size_t j = 0; j < count; j++) {
      printf("%d ", data[j]);
    }
  } else {
    fprintf(stderr, "Error: Datatype print not supported.");
  }
}

void print_buffers(const void *sbuf, const void *rbuf, const void *rbuf_gt,
                          size_t sbuf_count, size_t rbuf_count, MPI_Datatype dtype,
                          MPI_Comm comm, int use_barrier) {
  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  for(int i = 0; i < comm_sz; i++) {
    if(rank == i) {
      if (sbuf != NULL) {
        printf("\nRank %d:\nsendbuf: ", rank);
        print_buffer_helper(sbuf, sbuf_count, dtype);
        printf("\nrecvbuf: ");
      } else {
        printf("\nRank %d:\nrecvbuf: ", rank);
      }
      print_buffer_helper(rbuf, rbuf_count, dtype);
      printf("\ng_truth: ");
      print_buffer_helper(rbuf_gt, rbuf_count, dtype);
      printf("\n\n");
      fflush(stdout);
    }
    if (use_barrier == 0) MPI_Barrier(comm);
  }
}

#ifdef DEBUG
/**
 * @brief Little helper function to calculate integer powers for debugging purposes.
 *
 * @return base^power as an int64_t.
 */
static inline int64_t int_pow_64(int base, int exp) {
  int64_t result = 1, base_ = base;
  if(exp == 0) return 1;

  while (exp > 0) {
    if(exp % 2 == 1) result *= base_;
    base_ *= base_;
    exp /= 2;
  }
  return result;
}

/**
 * @brief Little helper function to calculate integer powers for debugging purposes.
 *
 * @return base^power as an int32_t.
 */
static inline int32_t int_pow_32(int base, int exp) {
  int32_t result = 1, base_ = base;
  if(exp == 0) return 1;

  while (exp > 0) {
    if(exp % 2 == 1) result *= base_;
    base_ *= base_;
    exp /= 2;
  }
  return result;
}

/**
 * @brief Little helper function to calculate integer powers for debugging purposes.
 *
 * @return base^power as an int.
 */
static inline int int_pow(int base, int exp) {
  int result = 1;
  if(exp == 0) return 1;

  while (exp > 0) {
    if(exp % 2 == 1) result *= base;
    base *= base;
    exp /= 2;
  }
  return result;
}

int debug_sbuf_generator(void *sbuf, MPI_Datatype dtype, size_t count,
                         MPI_Comm comm, test_routine_t test_routine) {
  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);

  // For BCAST and SCATTER, only rank 0 generates the sendbuf
  if((test_routine.collective == BCAST || test_routine.collective == SCATTER)
      && rank != 0) {
    return 0;
  }

  size_t real_sbuf_count =
    (test_routine.collective == ALLGATHER ||
     test_routine.collective == GATHER    ||
     test_routine.collective == ALLTOALL) ?
                                        count / (size_t) comm_sz : count;

  for(int i=0; i< real_sbuf_count; i++){
    if(dtype == MPI_INT64_T) {
      ((int64_t*)sbuf)[i] = int_pow_64(10, rank);
    } else if(dtype == MPI_INT32_T) {
      ((int32_t*)sbuf)[i] = int_pow_32(10, rank);
    } else if(dtype == MPI_INT){
      ((int*)sbuf)[i] = int_pow(10, rank);
    } else {
      fprintf(stderr, "Error: Datatype not implemented for `debug_sbuf_init`...");
      return -1;
    }
  }
  return 0;
}

void debug_print_buffers(const void *rbuf, const void *rbuf_gt, size_t count,
                        MPI_Datatype dtype, MPI_Comm comm, int use_barrier) {
  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  for(int i = 0; i < comm_sz; i++) {
    if(rank == i) {
      printf("\nRank %d:\nrecvbuf: ", rank);
      print_buffer_helper(rbuf, count, dtype);
      printf("\ng_truth: ");
      print_buffer_helper(rbuf_gt, count, dtype);
      printf("\n\n");
    }
    fflush(stdout);
    if(use_barrier == 0) MPI_Barrier(comm);
  }
}

#endif // DEBUG
