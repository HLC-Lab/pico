#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "bench_utils.h"
#include "libswing.h"

int main(int argc, char *argv[]) {
  MPI_Init(NULL, NULL);
  MPI_Comm comm = MPI_COMM_WORLD;
  // MPI_Comm inter_comm, intra_comm;
  MPI_Datatype dtype;
#ifdef CUDA_AWARE
  cudaError_t err;
  void *d_sbuf = NULL, *d_rbuf = NULL, *d_rbuf_gt = NULL;
#endif
  int rank, comm_sz, line, iter;
  size_t count, type_size;
  void *sbuf = NULL, *rbuf = NULL, *rbuf_gt = NULL;
  double *times = NULL, *all_times = NULL, *highest = NULL;
  const char *algorithm, *type_string; //, *is_hier = getenv("HIERARCHICAL");
  test_routine_t test_routine;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);

  // TODO: Continue with hierarchical communicator setup
  // if (is_hier == NULL) { line = __LINE__; goto err_hndl; }
  // if (strcmp(is_hier, "yes") == 0) {
  //   if (split_communicator(&inter_comm, &intra_comm) != MPI_SUCCESS) {
  //     line = __LINE__; goto err_hndl;
  //   }
  // }

  // Get test arguments
  if(get_command_line_arguments(argc, argv, &count, &iter, &algorithm, &type_string) == -1 ||
      get_routine (&test_routine, algorithm) == -1 ||
      get_data_type(type_string, &dtype, &type_size) == -1 ){
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

#ifdef CUDA_AWARE
  if(test_routine.allocator_cuda(&d_sbuf, &d_rbuf, &d_rbuf_gt, count, type_size, comm) != 0){
    line = __LINE__;
    goto err_hndl;
  }
#endif // CUDA_AWARE

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

#ifdef CUDA_AWARE
  if (coll_memcpy_host_to_device(&d_sbuf, &sbuf, count, type_size, test_routine.collective) != 0){
    line = __LINE__;
    goto err_hndl;
  }

  void *tmpsbuf = sbuf;
  void *tmprbuf = rbuf;
  sbuf = d_sbuf;
  rbuf = d_rbuf;
#endif

  // Perform the test based on the collective type and algorithm
  // The test is performed iter times
  if(test_loop(test_routine, sbuf, rbuf, count, dtype, comm, iter, times) != 0){
    line = __LINE__;
    goto err_hndl;
  }

#ifdef CUDA_AWARE
  rbuf = tmprbuf;
  sbuf = tmpsbuf;
  if (coll_memcpy_device_to_host(&d_rbuf, &rbuf, count, type_size, test_routine.collective) != 0){
    line = __LINE__;
    goto err_hndl;
  }
#endif

  // Check the results against the ground truth
  if(ground_truth_check(test_routine, sbuf, rbuf, rbuf_gt, count, dtype, comm) != 0){
    line = __LINE__;
    goto err_hndl;
  }

#ifndef DEBUG
  // Gather all process times to rank 0 and find the highest execution time of each iteration
  PMPI_Gather(times, iter, MPI_DOUBLE, all_times, iter, MPI_DOUBLE, 0, comm);

  if(test_routine.collective != REDUCE) {
    PMPI_Reduce(times, highest, iter, MPI_DOUBLE, MPI_MAX, 0, comm);
  } else {
    // Use custom reduce since you can have iter < comm_sz (it can crash for rabenseifner type reduce)
    reduce_swing_lat(times, highest, (size_t) iter, MPI_DOUBLE, MPI_MAX, 0, comm);
  }

  if (rank == 0) {
    if (swing_allreduce_segsize != 0) {
      printf("-------------------------------------------------------------------------------------------------------------------\n");
      printf("   %-30s\n    Last Iter Time: %15" PRId64"ns     %10ld elements of %s dtype\t%6d iter\t%8ld segsize\n", algorithm, (int64_t) (highest[iter-1] * 1e9), count, type_string, iter, swing_allreduce_segsize);
    } else {
      printf("-----------------------------------------------------------------------------------------------\n");
      printf("   %-30s\n    Last Iter Time: %15" PRId64"ns     %10ld elements of %s dtype\t%6d iter\n", algorithm, (int64_t) (highest[iter-1] * 1e9), count, type_string, iter);
    }
  }
  
  // Save results to a .csv file inside `/data/` subdirectory. Bash script `run_test_suite.sh`
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
#endif // DEBUG

  // Clean up
  if(NULL != sbuf)    free(sbuf);
  if(NULL != rbuf)    free(rbuf);
  if(NULL != rbuf_gt) free(rbuf_gt);
  free(times);

  if(rank == 0) {
    free(all_times);
    free(highest);
  }

#ifdef CUDA_AWARE
  if(NULL != d_sbuf)    cudaFree(d_sbuf);
  if(NULL != d_rbuf)    cudaFree(d_rbuf);
  if(NULL != d_rbuf_gt) cudaFree(d_rbuf_gt);
#endif // CUDA_AWARE

  MPI_Barrier(comm);

  MPI_Finalize();

  return EXIT_SUCCESS;

err_hndl:
  fprintf(stderr, "\n%s: line %d\tError invoked by rank %d\n\n", __FILE__, line, rank);
  (void)line;  // silence compiler warning

  if(NULL != sbuf)    free(sbuf);
  if(NULL != rbuf)    free(rbuf);
  if(NULL != rbuf_gt) free(rbuf_gt);
  if(NULL != times)   free(times);

  if(rank == 0) {
    if(NULL != all_times)  free(all_times);
    if(NULL != highest)    free(highest);
  }

#ifdef CUDA_AWARE
  if(NULL != d_sbuf)    cudaFree(d_sbuf);
  if(NULL != d_rbuf)    cudaFree(d_rbuf);
  if(NULL != d_rbuf_gt) cudaFree(d_rbuf_gt);
#endif // CUDA_AWARE

  MPI_Abort(comm, MPI_ERR_UNKNOWN);

  return EXIT_FAILURE;
}

