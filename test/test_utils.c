#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>

#include "test_utils.h"


int rand_array_generator(void *target, const char *type_string, size_t array_size, int rank) {
  unsigned int seed = time(NULL) + rank;

  if (strcmp(type_string, "int8") == 0) {
    for (size_t i = 0; i < array_size; i++) {
      ((int8_t *)target)[i] = (int8_t)(rand_r(&seed) % 256) - 128;
    }
  } else if (strcmp(type_string, "int16") == 0) {
    for (size_t i = 0; i < array_size; i++) {
      ((int16_t *)target)[i] = (int16_t)(rand_r(&seed) % 65536) - 32768;
    }
  } else if (strcmp(type_string, "int32") == 0) {
    for (size_t i = 0; i < array_size; i++) {
      ((int32_t *)target)[i] = (int32_t)(rand_r(&seed));
    }
  } else if (strcmp(type_string, "int64") == 0) {
    for (size_t i = 0; i < array_size; i++) {
      ((int64_t *)target)[i] = ((int64_t)rand_r(&seed) << 32) | rand_r(&seed);
    }
  } else if (strcmp(type_string, "int") == 0) {
    for (size_t i = 0; i < array_size; i++) {
      ((int *)target)[i] = (int)rand_r(&seed);
    }
  } else if (strcmp(type_string, "float") == 0) {
    for (size_t i = 0; i < array_size; i++) {
      ((float *)target)[i] = (float)rand_r(&seed) / RAND_MAX * 100.0f;
    }
  } else if (strcmp(type_string, "double") == 0) {
    for (size_t i = 0; i < array_size; i++) {
      ((double *)target)[i] = (double)rand_r(&seed) / RAND_MAX * 100.0;
    }
  } else if (strcmp(type_string, "char") == 0) {
    for (size_t i = 0; i < array_size; i++) {
      ((char *)target)[i] = (char)(rand_r(&seed) % 256);
    }
  } else {
    fprintf(stderr, "Error: sendbuf not generated correctly. Aborting...\n");
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

int are_equal_eps(const void *buf_1, const void *buf_2, size_t buffer_count, const MPI_Datatype dtype, int comm_sz) {
  if (buffer_count == 0) return 0;

  size_t i;

  if (MPI_FLOAT == dtype) {
    float *b1 = (float *) buf_1;
    float *b2 = (float *) buf_2;

    float epsilon = comm_sz * TEST_BASE_EPSILON_FLOAT * 100.0f;

    for (i = 0; i < buffer_count; i++) {
      if (fabs(b1[i] - b2[i]) > epsilon) {
        return -1;
      }
    }
  } else if (MPI_DOUBLE == dtype) {
    double *b1 = (double *) buf_1;
    double *b2 = (double *) buf_2;

    double epsilon = comm_sz * TEST_BASE_EPSILON_DOUBLE * 100.0;

    for (i = 0; i < buffer_count; i++) {
      if (fabs(b1[i] - b2[i]) > epsilon) {
        return -1;
      }
    }
  }

  return 0;
}

