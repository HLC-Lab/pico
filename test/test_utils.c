#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>

#include "test_utils.h"


int rand_sbuf_generator(void *sbuf, const char *type_string, size_t count,
                         MPI_Comm comm, routine_decision_t test_routine) {
  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);
  unsigned int seed = time(NULL) + rank;
  
  // If generating sendbuf for an ALLGATHER, the number of element is
  // count / comm_sz
  size_t real_sbuf_count = (test_routine.collective == ALLGATHER) ?
                                        count / (size_t) comm_sz : count;

  if (strcmp(type_string, "int8") == 0) {
    for (size_t i = 0; i < real_sbuf_count; i++) {
      ((int8_t *)sbuf)[i] = (int8_t)(rand_r(&seed) % 256) - 128;
    }
  } else if (strcmp(type_string, "int16") == 0) {
    for (size_t i = 0; i < real_sbuf_count; i++) {
      ((int16_t *)sbuf)[i] = (int16_t)(rand_r(&seed) % 65536) - 32768;
    }
  } else if (strcmp(type_string, "int32") == 0) {
    for (size_t i = 0; i < real_sbuf_count; i++) {
      ((int32_t *)sbuf)[i] = (int32_t)(rand_r(&seed));
    }
  } else if (strcmp(type_string, "int64") == 0) {
    for (size_t i = 0; i < real_sbuf_count; i++) {
      ((int64_t *)sbuf)[i] = ((int64_t)rand_r(&seed) << 32) | rand_r(&seed);
    }
  } else if (strcmp(type_string, "int") == 0) {
    for (size_t i = 0; i < real_sbuf_count; i++) {
      ((int *)sbuf)[i] = (int)rand_r(&seed);
    }
  } else if (strcmp(type_string, "float") == 0) {
    for (size_t i = 0; i < real_sbuf_count; i++) {
      ((float *)sbuf)[i] = (float)rand_r(&seed) / RAND_MAX * 100.0f;
    }
  } else if (strcmp(type_string, "double") == 0) {
    for (size_t i = 0; i < real_sbuf_count; i++) {
      ((double *)sbuf)[i] = (double)rand_r(&seed) / RAND_MAX * 100.0;
    }
  } else if (strcmp(type_string, "char") == 0) {
    for (size_t i = 0; i < real_sbuf_count; i++) {
      ((char *)sbuf)[i] = (char)(rand_r(&seed) % 256);
    }
  } else {
    fprintf(stderr, "Error: sbuf not generated correctly. Aborting...\n");
    return -1;
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
static inline int64_t int_pow(int base, int exp) {
  int result = 1;
  while (exp > 0) {
    if (exp % 2 == 1) result *= base;
    base *= base;
    exp /= 2;
  }
  return (int64_t) result;
}

void debug_sbuf_init(void *sbuf, MPI_Comm comm) {
  int rank, comm_sz;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_sz);

  for(int i=0; i<comm_sz; i++){
    ((int64_t*)sbuf)[i] = int_pow(10, rank);
  }
}
#endif

