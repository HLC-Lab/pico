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
  MPI_Datatype dtype;
  int rank, comm_sz, line, iter;
  size_t count, type_size;
  void *sbuf = NULL, *rbuf = NULL, *rbuf_gt = NULL;
  double *times = NULL, *all_times = NULL, *highest = NULL;
  const char *algorithm, *type_string;
  test_routine_t test_routine;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);


  // Get test arguments
#ifndef DEBUG
  const char *outputdir = NULL, *data_dir = NULL, *output_level = NULL;
  outputdir = getenv("OUTPUT_DIR");
  data_dir = getenv("DATA_DIR");
  output_level = getenv("OUTPUT_LEVEL");
  if(outputdir == NULL || data_dir == NULL || output_level == NULL) {
    fprintf(stderr, "Error: Environment variables OUTPUT_DIR, DATA_DIR or OUTPUT_LEVEL not set. Aborting...");
    line = __LINE__;
    goto err_hndl;
  }
#endif // DEBUG
  if(get_command_line_arguments(argc, argv, &count, &iter, &algorithm, &type_string) == -1 ||
      get_routine (&test_routine, algorithm) == -1 ||
      get_data_type(type_string, &dtype, &type_size) == -1 ){
    line = __LINE__;
    goto err_hndl;
  }

  // Allocate memory for the buffers based on the collective type
  if(test_routine.allocator(&sbuf, &rbuf, &rbuf_gt, count, type_size, comm) != 0){
    line = __LINE__;
    goto err_hndl;
  }

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
  
  #ifndef DEBUG
  // Randomly generate the sbuf
  if(rand_sbuf_generator(sbuf, dtype, count, comm, test_routine) != 0){
    line = __LINE__;
    goto err_hndl;
  }
  #else
  // Initialize the sbuf with a sequence of powers of 10
  // WARNING: Only int32, int64 and int supported
  if(debug_sbuf_generator(sbuf, dtype, count, comm, test_routine) != 0){
    line = __LINE__;
    goto err_hndl;
  }
  #endif // DEBUG
  
  // Perform the test based on the collective type and algorithm
  // The test is performed iter times
  if(test_loop(test_routine, sbuf, rbuf, count, dtype, comm, iter, times) != 0){
    line = __LINE__;
    goto err_hndl;
  }

  // Check the results against the ground truth
  if(ground_truth_check(test_routine, sbuf, rbuf, rbuf_gt, count, dtype, comm) != 0){
    line = __LINE__;
    goto err_hndl;
  }

  #ifndef DEBUG
  // Gather all process times to rank 0 and find the highest execution time of each iteration
  PMPI_Gather(times, iter, MPI_DOUBLE, all_times, iter, MPI_DOUBLE, 0, comm);
  PMPI_Reduce(times, highest, iter, MPI_DOUBLE, MPI_MAX, 0, comm);

  if (rank == 0) {
    printf("--------------------------------------------------------------------------------------------\n");
    printf("   %-30s\n    Last Iter Time: %15" PRId64"ns     %10ld elements of %s dtype\t%6d iter\n", algorithm, (int64_t) (highest[iter-1] * 1e9), count, type_string, iter);
  }
  
  // Save results to a .csv file inside `/data/` subdirectory. Bash script `run_test_suite.sh`
  // is responsible to create the `/data/` subdir.
  if(rank == 0){
    char data_filename[128], data_fullpath[BENCH_MAX_PATH_LENGTH];
    snprintf(data_filename, sizeof(data_filename), "/%ld_%s_%s.csv",
             count, algorithm, type_string);
    if(concatenate_path(data_dir, data_filename, data_fullpath) == -1) {
      line = __LINE__;
      goto err_hndl;
    }
    if(write_output_to_file(output_level, data_fullpath, highest, all_times, iter) == -1){
      line = __LINE__;
      goto err_hndl;
    }
  }

  // Save to file allocations (it uses MPI parallel I/O operations)
  char alloc_filename[128] = "alloc.csv";
  char alloc_fullpath[BENCH_MAX_PATH_LENGTH];
  if(concatenate_path(outputdir, alloc_filename, alloc_fullpath) == -1){
    line = __LINE__;
    goto err_hndl;
  }

  // Write current allocations if and only if the file `alloc_fullpath`
  // does not exists (i.e. only the first time for each srun this function
  // is encountered the allocations will be registerd)
  int should_write_alloc = 0;
  if(rank == 0){
    should_write_alloc = file_not_exists(alloc_fullpath);
  }
  PMPI_Bcast(&should_write_alloc, 1, MPI_INT, 0, comm);
  if((should_write_alloc == 1) &&
      (write_allocations_to_file(alloc_fullpath, comm) != MPI_SUCCESS)){
    // Remove the file if the write operation failed
    if(rank == 0){ remove(alloc_fullpath); }
    line = __LINE__;
    goto err_hndl;
  }
  #endif // DEBUG

  // Clean up
  free(sbuf);
  if(NULL != rbuf) free(rbuf); // rbuf can be NULL if the test routine does not use it (e.g. Bcast)
  free(rbuf_gt);
  free(times);

  if(rank == 0) {
    free(all_times);
    free(highest);
  }

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

  MPI_Abort(comm, MPI_ERR_UNKNOWN);

  return EXIT_FAILURE;
}

