/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "pico_core_utils.h"
#include "pico_mpi_nccl_mapper.h"
#include "libpico.h"

int main(int argc, char *argv[]) {
  MPI_Init(NULL, NULL);

  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, comm_sz, line;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);

#if defined PICO_MPI_CUDA_AWARE || defined PICO_NCCL
  int gpus_count, gpu_selector;
  cudaError_t cuda_err;
  PICO_CORE_CUDA_CHECK(cudaGetDeviceCount(&gpus_count), cuda_err);

  if (gpus_count > 1) {
    gpu_selector = rank % gpus_count;

    PICO_CORE_CUDA_CHECK(cudaSetDevice(gpu_selector), cuda_err);

    MPI_Barrier(comm);
  }
#endif

  MPI_Datatype dtype;
  PICO_DTYPE_T loop_dtype; // Used only in test loop, other routines use mpi datatype
  int iter;
  size_t count, type_size;
  void *sbuf = NULL, *rbuf = NULL, *rbuf_gt = NULL;
  double *times = NULL, *all_times = NULL, *highest = NULL;
  const char *algorithm, *type_string; //, *is_hier = getenv("HIERARCHICAL");
  test_routine_t test_routine;

#if defined PICO_MPI_CUDA_AWARE || defined PICO_NCCL
  void *d_sbuf = NULL, *d_rbuf = NULL, *d_rbuf_gt = NULL;
#endif // PICO_MPI_CUDA_AWARE || PICO_NCCL
#ifdef PICO_NCCL
  ncclComm_t   nccl_comm;
  cudaStream_t stream;
  if (pico_nccl_init(comm, rank, comm_sz, &nccl_comm, &stream) == -1) {
    line = __LINE__;
    goto err_hndl;
  }
#endif // PICO_NCCL
#if defined PICO_INSTRUMENT && !defined PICO_NCCL
  int num_tags;
  const char **tag_names = NULL;
  double** tag_times = NULL;
#endif

#if defined PICO_INSTRUMENT && !defined PICO_NCCL && !defined PICO_MPI_CUDA_AWARE
#endif

  // TODO: Continue with hierarchical communicator setup
  // MPI_Comm inter_comm, intra_comm;
  // if (is_hier == NULL) { line = __LINE__; goto err_hndl; }
  // if (strcmp(is_hier, "yes") == 0) {
  //   if (split_communicator(&inter_comm, &intra_comm) != MPI_SUCCESS) {
  //     line = __LINE__; goto err_hndl;
  //   }
  // }

  // Get test arguments
  if(get_command_line_arguments(argc, argv, &count, &iter, &algorithm, &type_string) == -1 ||
      get_routine (&test_routine, algorithm) == -1 ||
      get_data_type(type_string, &loop_dtype, &dtype, &type_size) == -1 ){
    line = __LINE__;
    goto err_hndl;
  }

#ifndef DEBUG
  if (get_data_saving_options(&test_routine, count, algorithm, type_string) == -1) {
    line = __LINE__;
    goto err_hndl;
  }
#endif // DEBUG

  // Allocate memory for the buffers based on the collective type
  if(test_routine.allocator(&sbuf, &rbuf, &rbuf_gt, count, type_size, comm) != 0){
    line = __LINE__;
    goto err_hndl;
  }

#if defined PICO_MPI_CUDA_AWARE || defined PICO_NCCL
  if(test_routine.allocator_cuda(&d_sbuf, &d_rbuf, &d_rbuf_gt, count, type_size, comm) != 0){
    line = __LINE__;
    goto err_hndl;
  }
#endif // PICO_MPI_CUDA_AWARE || PICO_NCCL

  // Allocate memory for buffers independent of collective type
  times = (double *)calloc(iter, sizeof(double));
  if(rank == 0) {
    all_times = (double *)malloc(comm_sz * iter * sizeof(double));
    highest = (double *)malloc(iter * sizeof(double));
  }
  if(times == NULL || (rank == 0 && (all_times == NULL || highest == NULL))){
    fprintf(stderr, "Error: Memory allocation failed. Aborting...");
    line = __LINE__;
    goto err_hndl;
  }

#ifdef DEBUG
  // Initialize the sbuf with a sequence of powers of 10
  // WARNING: Only int32, int64 and int supported
  if(debug_sbuf_generator(sbuf, dtype, count, comm, test_routine) != 0){
    line = __LINE__;
    goto err_hndl;
  }
#else
  // Randomly generate the sbuf
  if(rand_sbuf_generator(sbuf, dtype, count, comm, test_routine) != 0){
    line = __LINE__;
    goto err_hndl;
  }
#endif // DEBUG

#if defined PICO_INSTRUMENT && !defined PICO_NCCL
  libpico_init_tags();
  #if defined PICO_MPI_CUDA_AWARE
  if (run_coll_once(test_routine, d_sbuf, d_rbuf, count, dtype, comm) != MPI_SUCCESS) {
    line = __LINE__;
    goto err_hndl;
  }
  #else
  if (run_coll_once(test_routine, sbuf, rbuf, count, dtype, comm) != MPI_SUCCESS) {
    line = __LINE__;
    goto err_hndl;
  }
  #endif
  num_tags = libpico_count_tags();
  if (num_tags <= 0) {
    fprintf(stderr, "Error: No tags were created. Aborting...");
    line = __LINE__;
    goto err_hndl;
  }

  tag_names = malloc(num_tags * sizeof(char *));
  if (tag_names == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...");
    line = __LINE__;
    goto err_hndl;
  }

  if (libpico_get_tag_names(tag_names, num_tags) != 0) {
    fprintf(stderr, "Error: Failed to get tag names. Aborting...");
    line = __LINE__;
    goto err_hndl;
  }
  
  tag_times = (double **) malloc(num_tags * sizeof(double *));
  if (tag_times == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...");
    line = __LINE__;
    goto err_hndl;
  }

  tag_times[0] = (double *) calloc(num_tags * iter, sizeof(double));
  if (tag_times[0] == NULL) {
    fprintf(stderr, "Error: Memory allocation failed. Aborting...");
    line = __LINE__;
    goto err_hndl;
  }

  for (int i = 1; i < num_tags; i++) {
    tag_times[i] = tag_times[0] + i * iter;
  }

  if (libpico_build_handles(tag_times, num_tags, iter) != 0) {
    fprintf(stderr, "Error: Failed to build handles. Aborting...");
    line = __LINE__;
    goto err_hndl;
  }

  if (libpico_clear_tags() != 0) {
    fprintf(stderr, "Error: Failed to clear tags. Aborting...");
    line = __LINE__;
    goto err_hndl;
  }
#endif


#if defined PICO_MPI_CUDA_AWARE || defined PICO_NCCL
  if (coll_memcpy_host_to_device(&d_sbuf, &sbuf, count, type_size, test_routine.collective) != 0){
    line = __LINE__;
    goto err_hndl;
  }

  void *tmpsbuf = sbuf;
  void *tmprbuf = rbuf;
  sbuf = d_sbuf;
  rbuf = d_rbuf;
#endif // PICO_MPI_CUDA_AWARE || PICO_NCCL

  // Perform the test based on the collective type and algorithm
  // The test is performed iter times
#ifndef PICO_NCCL
  if(test_loop(test_routine, sbuf, rbuf, count, loop_dtype, comm, iter, times) != 0){
#else
  if(test_loop(test_routine, sbuf, rbuf, count, loop_dtype, nccl_comm, stream, iter, times) != 0){
#endif // PICO_NCCL
    line = __LINE__;
    goto err_hndl;
  }

#if defined PICO_MPI_CUDA_AWARE || defined PICO_NCCL
  rbuf = tmprbuf;
  sbuf = tmpsbuf;
  if (coll_memcpy_device_to_host(&d_rbuf, &rbuf, count, type_size, test_routine.collective) != 0){
    line = __LINE__;
    goto err_hndl;
  }
#endif // PICO_MPI_CUDA_AWARE || PICO_NCCL

  // Check the results against the ground truth
  if(ground_truth_check(test_routine, sbuf, rbuf, rbuf_gt, count, dtype, comm) != 0){
    line = __LINE__;
    goto err_hndl;
  }

#ifndef DEBUG

#if !(defined PICO_INSTRUMENT && !defined PICO_NCCL)
  // Gather all process times to rank 0 and find the highest execution time of each iteration
  PMPI_Gather(times, iter, MPI_DOUBLE, all_times, iter, MPI_DOUBLE, 0, comm);

  if(test_routine.collective != REDUCE) {
    PMPI_Reduce(times, highest, iter, MPI_DOUBLE, MPI_MAX, 0, comm);
  } else {
    // Use custom reduce since you can have iter < comm_sz (it can crash for rabenseifner type reduce)
    reduce_bine_lat(times, highest, (size_t) iter, MPI_DOUBLE, MPI_MAX, 0, comm);
  }

  if (rank == 0) {
    if (bine_allreduce_segsize != 0) {
      printf("-------------------------------------------------------------------------------------------------------------------\n");
      printf("   %-30s\n    Last Iter Time: %15" PRId64"ns     %10ld elements of %s dtype\t%6d iter\t%8ld segsize\n",
             algorithm, (int64_t) (highest[iter-1] * 1e9), count, type_string, iter, bine_allreduce_segsize);
    } else {
      printf("-----------------------------------------------------------------------------------------------\n");
      printf("   %-30s\n    Last Iter Time: %15" PRId64"ns     %10ld elements of %s dtype\t%6d iter\n",
             algorithm, (int64_t) (highest[iter-1] * 1e9), count, type_string, iter);
    }
  }
  
  // Save results to a .csv file inside `/data/` subdirectory. Bash script `orchestrator.sh`
  // is responsible to create the `/data/` subdir.
  if(rank == 0){
    if(write_output_to_file(test_routine, highest, all_times, iter) == -1){
      line = __LINE__;
      goto err_hndl;
    }
  }

  // Write current allocations if and only if the file `alloc_fullpath`
  // does not exists
  int should_write_alloc = 0;
  if(rank == 0){
    should_write_alloc = file_not_exists(test_routine.alloc_file);
  }
  PMPI_Bcast(&should_write_alloc, 1, MPI_INT, 0, comm);
  if((should_write_alloc == 1) &&
      (write_allocations_to_file(test_routine.alloc_file, comm) != MPI_SUCCESS)){
    // Remove the file if the write operation failed
    if(rank == 0){ remove(test_routine.alloc_file); }
    line = __LINE__;
    goto err_hndl;
  }


#else
  if (rank == 0) {
    if (bine_allreduce_segsize != 0) {
      printf("-------------------------------------------------------------------------------------------------------------------\n");
      printf("-------------------------------------------------------------------------------------------------------------------\n");
      printf("   %-30s\n    Last Iter Time: %15" PRId64"ns     %10ld elements of %s dtype\t%6d iter\t%8ld segsize\n",
             algorithm, (int64_t) (times[iter-1] * 1e9), count, type_string, iter, bine_allreduce_segsize);
    } else {
      printf("-----------------------------------------------------------------------------------------------\n");
      printf("-----------------------------------------------------------------------------------------------\n");
      printf("   %-30s\n    Last Iter Time: %15" PRId64"ns     %10ld elements of %s dtype\t%6d iter\n",
             algorithm, (int64_t) (times[iter-1] * 1e9), count, type_string, iter);
    }
    const int name_w = pico_name_col_width(tag_names, num_tags, 20, LIBPICO_TAG_NAME_MAX);
    const int sep_w  = name_w + 2 + 12 + 2 + 8; /* name + ": " + 12ns + "  " + "xx.x%" */
    int64_t total_ns = (int64_t)(times[iter-1] * 1e9);
    if (total_ns < 0) total_ns = 0;

    /* header row */

    for (int i = 0; i < sep_w; ++i) putchar('-');
    printf("\n%-*s  %12s  %s\n", name_w, "Tag", "Time (ns)", "    %");
    for (int i = 0; i < sep_w; ++i) putchar('-');
    putchar('\n');

    /* rows */
    for (int t = 0; t < num_tags; ++t) {
        int64_t t_ns = (int64_t)(tag_times[t][iter-1] * 1e9);
        if (t_ns < 0) t_ns = 0;
        double pct = (total_ns > 0) ? (100.0 * (double)t_ns / (double)total_ns) : 0.0;

        /* tag name truncated to column width if too long */
        printf("%-*.*s  %12" PRId64 "  %6.1f%%\n",
               name_w, name_w, tag_names[t],
               t_ns, pct);
    }

    if(write_instrument_output_to_file(test_routine, times, tag_times, tag_names, iter) == -1){
      line = __LINE__;
      goto err_hndl;
    }
  }
#endif // !(defined PICO_INSTRUMENT && !defined PICO_NCCL)

#endif // DEBUG

  // Clean up
  if(NULL != sbuf && sbuf != rbuf)    free(sbuf);
  if(NULL != rbuf)                    free(rbuf);
  if(NULL != rbuf_gt)                 free(rbuf_gt);
  free(times);

  if(rank == 0) {
    free(all_times);
    free(highest);
  }

#if defined PICO_INSTRUMENT && !defined PICO_NCCL
  free(tag_names);
  free(tag_times[0]);
  free(tag_times);
#endif

#ifdef PICO_NCCL
  if(pico_nccl_finalize(nccl_comm, stream) != 0) {
    line = __LINE__;
    goto err_hndl;
  }
#endif
#if defined PICO_MPI_CUDA_AWARE || defined PICO_NCCL
  if(NULL != d_sbuf && d_sbuf != d_rbuf)    cudaFree(d_sbuf);
  if(NULL != d_rbuf)                        cudaFree(d_rbuf);
  if(NULL != d_rbuf_gt)                     cudaFree(d_rbuf_gt);
#endif // PICO_MPI_CUDA_AWARE || PICO_NCCL

  MPI_Barrier(comm);

  MPI_Finalize();

  return EXIT_SUCCESS;

err_hndl:
  fprintf(stderr, "\n%s: line %d\tError invoked by rank %d\n\n", __FILE__, line, rank);
  (void)line;  // silence compiler warning

  if(NULL != sbuf && sbuf != rbuf)    free(sbuf);
  if(NULL != rbuf)                    free(rbuf);
  if(NULL != rbuf_gt)                 free(rbuf_gt);
  if(NULL != times)                   free(times);

  if(rank == 0) {
    if(NULL != all_times)  free(all_times);
    if(NULL != highest)    free(highest);
  }

#if defined PICO_MPI_CUDA_AWARE || defined PICO_NCCL
  if(NULL != d_sbuf && d_sbuf != d_rbuf)    cudaFree(d_sbuf);
  if(NULL != d_rbuf)                        cudaFree(d_rbuf);
  if(NULL != d_rbuf_gt)                     cudaFree(d_rbuf_gt);
#endif // PICO_MPI_CUDA_AWARE || PICO_NCCL

#if defined PICO_INSTRUMENT && !defined PICO_NCCL
  if (NULL != tag_names) free(tag_names);
  if (NULL != tag_times) {
    if (NULL != tag_times[0]) free(tag_times[0]);
    free(tag_times);
  }
#endif

  MPI_Abort(comm, MPI_ERR_UNKNOWN);

  return EXIT_FAILURE;
}
