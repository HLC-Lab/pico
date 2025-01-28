#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <sys/stat.h>

#include "test_utils.h"


int write_output_to_file(const char *fullpath, double *highest, double *all_times, int iter) {
  int comm_sz;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  FILE *output_file = fopen(fullpath, "w");
  if (output_file == NULL) {
    fprintf(stderr, "Error: Opening file %s for writing\n", fullpath);
    return -1;
  }

  // Write the header with ranks from rank0 to rankN
  fprintf(output_file, "highest");
  for (int rank = 0; rank < comm_sz; rank++) {
    fprintf(output_file, ",rank%d", rank);
  }
  fprintf(output_file, "\n");

  // Write the timing data
  for (int i = 0; i < iter; i++) {
    fprintf(output_file, "%" PRId64, (int64_t)(highest[i] * 1e9));
    for (int j = 0; j < comm_sz; j++) {
      fprintf(output_file, ",%" PRId64, (int64_t)(all_times[j * iter + i] * 1e9));
    }
    fprintf(output_file, "\n");
  }

  fclose(output_file);
  return 0;
}


int file_not_exists(const char* filename) {
  struct stat buffer;
  return (stat(filename, &buffer) != 0) ? 1 : 0;
}


int write_allocations_to_file(const char* filename, MPI_Comm comm) {
  int rank, comm_sz;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;

  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  MPI_Get_processor_name(processor_name, &name_len);

  MPI_File file;
  if (MPI_File_open(comm, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file) != MPI_SUCCESS) {
    if (rank == 0) {
      fprintf(stderr, "Error: Opening file %s for writing\n", filename);
    }
    return MPI_ERR_FILE;
  }

  const char header[] = "MPI_Rank,allocation\n";
  // Rank 0 writes the header to the file
  if (rank == 0) {
    MPI_File_write_at(file, 0, header, sizeof(header) - 1, MPI_CHAR, MPI_STATUS_IGNORE);
  }

  MPI_Barrier(comm);  // Ensure header is written before writing rank data

  // Define a fixed-length buffer for each rank's entry
  char buffer[MPI_MAX_PROCESSOR_NAME + 16];  // Fixed space for rank, comma, name, and newline
  snprintf(buffer, sizeof(buffer), "%d,%s\n", rank, processor_name);

  // Calculate a unique offset for each rank using fixed-size entries
  MPI_Offset offset = sizeof(header) - 1 + rank * (MPI_MAX_PROCESSOR_NAME + 16);

  // Write each rank's data at its calculated offset
  MPI_File_write_at(file, offset, buffer, strlen(buffer), MPI_CHAR, MPI_STATUS_IGNORE);

  MPI_File_close(&file);
  return MPI_SUCCESS;
}


int rand_sbuf_generator(void *sbuf, MPI_Datatype dtype, size_t count,
                         MPI_Comm comm, routine_decision_t test_routine) {
  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  unsigned int seed = time(NULL) + rank;
  
  // If generating sendbuf for an ALLGATHER, the number of element is
  // count / comm_sz
  size_t real_sbuf_count =
    (test_routine.collective == ALLGATHER) ?
                                        count / (size_t) comm_sz : count;

  for (size_t i = 0; i < real_sbuf_count; i++) {
    if (dtype == MPI_INT8_T) {
      ((int8_t *)sbuf)[i] = (int8_t)((rand_r(&seed) % 256) - 128);
    } else if(dtype == MPI_INT16_T) {
      ((int16_t *)sbuf)[i] = (int16_t)((rand_r(&seed) % 65536) - 32768);
    } else if (dtype == MPI_INT32_T) {
      ((int32_t *)sbuf)[i] = (int32_t)(rand_r(&seed));
    } else if (dtype == MPI_INT64_T) {
      ((int64_t *)sbuf)[i] = ((int64_t)rand_r(&seed) << 32) | rand_r(&seed);
    } else if (dtype == MPI_INT) {
      ((int *)sbuf)[i] = (int)rand_r(&seed);
    } else if (dtype == MPI_FLOAT) {
      ((float *)sbuf)[i] = (float)rand_r(&seed) / RAND_MAX * 100.0f;
    } else if (dtype == MPI_DOUBLE) {
      ((double *)sbuf)[i] = (double)rand_r(&seed) / RAND_MAX * 100.0;
    } else if (dtype == MPI_CHAR) {
      ((char *)sbuf)[i] = (char)((rand_r(&seed) % 256) - 128);
    } else if (dtype == MPI_UNSIGNED_CHAR) {
      ((unsigned char *)sbuf)[i] = (unsigned char)(rand_r(&seed) % 256);
    } else {
      fprintf(stderr, "Error: sbuf not generated correctly. Aborting...\n");
      return -1;
    }
  }

  return 0;
}


int concatenate_path(const char *dir_path, const char *filename, char *fullpath) {
  if (dir_path == NULL || filename == NULL) {
    fprintf(stderr, "Directory path or filename is NULL.\n");
    return -1;
  }

  size_t dir_path_len = strlen(dir_path);
  size_t filename_len = strlen(filename);

  if (dir_path_len == 0) {
    fprintf(stderr, "Directory path is empty.\n");
    return -1;
  }

  if (dir_path_len + filename_len + 2 > TEST_MAX_PATH_LENGTH) {
    fprintf(stderr, "Combined path length exceeds buffer size.\n");
    return -1;
  }

  strcpy(fullpath, dir_path);
  if (dir_path[dir_path_len - 1] != '/') {
    strcat(fullpath, "/");
  }
  strcat(fullpath, filename);

  return 0;
}


int are_equal_eps(const void *buf_1, const void *buf_2, size_t count,
                  MPI_Datatype dtype, int comm_sz) {
  if (count == 0) return 0;

  size_t i;

  if (MPI_FLOAT == dtype) {
    float *b1 = (float *) buf_1;
    float *b2 = (float *) buf_2;

    float epsilon = comm_sz * TEST_BASE_EPSILON_FLOAT * 100.0f;

    for (i = 0; i < count; i++) {
      if (fabs(b1[i] - b2[i]) > epsilon) {
        return -1;
      }
    }
  } else if (MPI_DOUBLE == dtype) {
    double *b1 = (double *) buf_1;
    double *b2 = (double *) buf_2;

    double epsilon = comm_sz * TEST_BASE_EPSILON_DOUBLE * 100.0;

    for (i = 0; i < count; i++) {
      if (fabs(b1[i] - b2[i]) > epsilon) {
        return -1;
      }
    }
  }

  return 0;
}

#ifdef DEBUG
/**
 * @brief Little helper function to calculate integer powers for debugging purposes.
 *
 * @return base^power as an int64_t.
 */
static inline int64_t int_pow_64(int base, int exp) {
  int64_t result = 1, base_ = base;
  if (exp == 0) return 1;

  while (exp > 0) {
    if (exp % 2 == 1) result *= base_;
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
  if (exp == 0) return 1;

  while (exp > 0) {
    if (exp % 2 == 1) result *= base_;
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
  if (exp == 0) return 1;

  while (exp > 0) {
    if (exp % 2 == 1) result *= base;
    base *= base;
    exp /= 2;
  }
  return result;
}

int debug_sbuf_generator(void *sbuf, MPI_Datatype dtype, size_t count,
                         MPI_Comm comm, routine_decision_t test_routine) {
  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);

  size_t real_sbuf_count =
    (test_routine.collective == ALLGATHER) ?
                                        count / (size_t) comm_sz : count;

  for(int i=0; i< real_sbuf_count; i++){
    if (dtype == MPI_INT64_T) {
      ((int64_t*)sbuf)[i] = int_pow_64(10, rank);
    } else if (dtype == MPI_INT32_T) {
      ((int32_t*)sbuf)[i] = int_pow_32(10, rank);
    } else if (dtype == MPI_INT){
      ((int*)sbuf)[i] = int_pow(10, rank);
    } else {
      fprintf(stderr, "Error: Datatype not implemented for `debug_sbuf_init`...\n");
      return -1;
    }
  }
  return 0;
}


void debug_print_buffers(void *rbuf, void *rbuf_gt, size_t count,
                         MPI_Datatype dtype, MPI_Comm comm){
  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);

  for (int i = 0; i < comm_sz; i++) {
    if (rank == i) {
      printf("Rank %d:\n rbuf_gt \t rbuf\n", rank);
      for (size_t j = 0; j < count; j++) {
        // I tried with a switch statement, but in OMPI datatypes are pointer
        // to structures, so I can't use them in a switch statement.
        if (dtype == MPI_INT64_T){
          printf("%" PRId64 "\t\t %" PRId64"\n", ((int64_t *)rbuf_gt)[j], ((int64_t *)rbuf)[j]);
        } else if (dtype == MPI_INT32_T){
          printf("%" PRId32 "\t\t %" PRId32"\n", ((int32_t *)rbuf_gt)[j], ((int32_t *)rbuf)[j]);
        } else if (dtype == MPI_INT16_T){
          printf("%" PRId16 "\t\t %" PRId16"\n", ((int16_t *)rbuf_gt)[j], ((int16_t *)rbuf)[j]);
        } else if (dtype == MPI_INT8_T){
          printf("%" PRId8 "\t\t %" PRId8"\n", ((int8_t *)rbuf_gt)[j], ((int8_t *)rbuf)[j]);
        } else if (dtype == MPI_INT){
          printf("%d\t\t %d\n", ((int *)rbuf_gt)[j], ((int *)rbuf)[j]);
        } else if (dtype == MPI_FLOAT){
          printf("%f\t\t %f\n", ((float *)rbuf_gt)[j], ((float *)rbuf)[j]);
        } else if (dtype == MPI_DOUBLE){
          printf("%f\t\t %f\n", ((double *)rbuf_gt)[j], ((double *)rbuf)[j]);
        } else {
          fprintf(stderr, "Error: Datatype print not supported.\n");
          break;
        }
      }
      printf("\n");
      fflush(stdout);
    }
    MPI_Barrier(comm);
  }
}

#endif
