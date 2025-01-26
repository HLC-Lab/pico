#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_utils.h"
#include "libswing.h"


int main(int argc, char *argv[]) {
  MPI_Init(NULL, NULL);
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, comm_sz, line;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);

  // Declare beforehand the buffers to allow for correct
  // error handling with goto
  void *sbuf = NULL, *rbuf = NULL, *rbuf_gt = NULL;
  double *times = NULL, *all_times = NULL, *highest = NULL;

  size_t count;
  int iter, algorithm;
  const char* type_string, *outputdir;
  // Get command line arguments
  if (get_command_line_arguments(argc, argv, &count, &iter,
                       &type_string, &algorithm, &outputdir) == -1){
    line = __LINE__;
    goto err_hndl;
  }
  
  routine_decision_t test_routine;
  if (get_routine(&test_routine, algorithm) == -1){
    line = __LINE__;
    goto err_hndl;
  }
  
  // Get size and MPI_Datatype from input `type_string`
  MPI_Datatype dtype;
  size_t type_size;
  if (get_data_type(type_string, &dtype, &type_size) == -1){
    line = __LINE__;
    goto err_hndl;
  }
  
  allocator_func_ptr allocator = get_allocator(test_routine);
  if (NULL == allocator){
    fprintf(stderr, "Error: allocator is NULL. Aborting...\n");
    line = __LINE__;
    goto err_hndl;
  }
  
  // Allocate memory for the buffers based on the collective type
  if (allocator(&sbuf, &rbuf, &rbuf_gt, count, type_size, comm) != 0){
    line = __LINE__;
    goto err_hndl;
  }

  // Allocate memory for buffers independent of collective type
  times = (double *)malloc(iter * sizeof(double));
  if (times == NULL){
    fprintf(stderr, "Error: Memory allocation failed. Aborting...\n");
    line = __LINE__;
    goto err_hndl;
  }
  if (rank == 0) {
    all_times = (double *)malloc(comm_sz * iter * sizeof(double));
    highest = (double *)malloc(iter * sizeof(double));
    if (all_times == NULL || highest == NULL){
      fprintf(stderr, "Error: Memory allocation failed. Aborting...\n");
      line = __LINE__;
      goto err_hndl;
    }
  }

  // randomly generate the sbuf
  if (rand_sbuf_generator(sbuf, type_string, count, comm, test_routine) == -1){
    line = __LINE__;
    goto err_hndl;
  }

  switch (test_routine.collective){
    case ALLREDUCE:
      // Perform the test benchmark for `iter` iterations
      allreduce_test_loop(sbuf, rbuf, count, dtype, MPI_SUM, comm, iter,
                          times, test_routine.algorithm.allreduce_algorithm);

      // Do a ground-truth check on the correctness of last iteration result
      if (allreduce_gt_check(sbuf, rbuf, count, dtype, MPI_SUM, comm, rbuf_gt) != 0){
        line = __LINE__;
        goto err_hndl;
      }
      break;
    case ALLGATHER:
      // Perform the test benchmark for `iter` iterations
      allgather_test_loop(sbuf, count / (size_t) comm_sz, dtype,
                          rbuf, count / (size_t) comm_sz, dtype,
                          comm, iter, times,
                          test_routine.algorithm.allgather_algorithm);

      // Do a ground-truth check on the correctness of last iteration result
      if (allgather_gt_check(sbuf, count / (size_t) comm_sz, dtype,
                             rbuf, count / (size_t) comm_sz, dtype,
                             comm, rbuf_gt) != 0){
        line = __LINE__;
        goto err_hndl;
      }
      break;
    default:
      fprintf(stderr, "still not implemented, aborting...\n");
      line = __LINE__;
      goto err_hndl;
  }


  // Gather all process times to rank 0 and find the highest execution time of each iteration
  PMPI_Gather(times, iter, MPI_DOUBLE, all_times, iter, MPI_DOUBLE, 0, comm);
  PMPI_Reduce(times, highest, iter, MPI_DOUBLE, MPI_MAX, 0, comm);
  
  // Save results to a .csv file inside `/data/` subdirectory. Bash script `run_test_suite.sh`
  // is responsible to create the `/data/` subdir.
  if (rank == 0){
    char data_filename[128], data_fullpath[TEST_MAX_PATH_LENGTH];
    snprintf(data_filename, sizeof(data_filename), "data/%d_%ld_%s_%d.csv",
             comm_sz, count, type_string, (int) algorithm);
    if (concatenate_path(outputdir, data_filename, data_fullpath) == -1) {
      fprintf(stderr, "Error: Failed to create fullpath.\n");
      line = __LINE__;
      goto err_hndl;
    }
    if (write_output_to_file(data_fullpath, highest, all_times, iter) == -1){
      line = __LINE__;
      goto err_hndl;
    }
  }

  // Save to file allocations (it uses MPI parallel I/O operations)
  char alloc_filename[128] = "alloc.csv";
  char alloc_fullpath[TEST_MAX_PATH_LENGTH];
  if (concatenate_path(outputdir, alloc_filename, alloc_fullpath) == -1){
    fprintf(stderr, "Error: Failed to create alloc_fullpath.\n");
    line = __LINE__;
    goto err_hndl;
  }

  // Write current allocations if and only if the file `alloc_fullpath`
  // does not exists (i.e. only the first time for each srun this function
  // is encountered the allocations will be registerd)
  int should_write_alloc;
  if (rank == 0){
    should_write_alloc = file_not_exists(alloc_fullpath);
  }
  PMPI_Bcast(&should_write_alloc, 1, MPI_INT, 0, comm);
  if (should_write_alloc == 1 && write_allocations_to_file(alloc_fullpath, comm) != MPI_SUCCESS){
    // TODO: delete the alloc file if it was created
    line = __LINE__;
    goto err_hndl;
  }

  // Clean up
  free(sbuf);
  free(rbuf);
  free(rbuf_gt);
  free(times);

  if (rank == 0) {
    free(all_times);
    free(highest);
  }

  MPI_Finalize();

  return EXIT_SUCCESS;

err_hndl:
  fprintf(stderr, "%s:%4d\tRank %d\n", __FILE__, line, rank);
  (void)line;  // silence compiler warning

  if (NULL != sbuf)    free(sbuf);
  if (NULL != rbuf)    free(rbuf);
  if (NULL != rbuf_gt) free(rbuf_gt);
  if (NULL != times)   free(times);

  if (rank == 0) {
    if (NULL != all_times)  free(all_times);
    if (NULL != highest)    free(highest);
  }

  MPI_Abort(comm, MPI_ERR_UNKNOWN);

  return EXIT_FAILURE;
}

