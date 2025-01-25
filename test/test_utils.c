#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <inttypes.h>
#include <sys/stat.h>

#include "test_utils.h"
#include "libswing.h"

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

/**
 * @brief Retrieves the MPI datatype and size based on a string identifier utilizing `type_map`.
 *
 * @param type_string String representation of the data type.
 * @param[out] dtype MPI datatype corresponding to the string.
 * @param[out] type_size Size of the datatype in bytes.
 * @return 0 on success, -1 if the data type is invalid.
 */
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

/**
 * @brief Parses command-line arguments and extracts parameters.
 *
 * @param argc Number of arguments.
 * @param argv Argument vector.
 * @param[out] array_size Size of the array.
 * @param[out] iter Number of iterations.
 * @param[out] type_string Data type as a string.
 * @param[out] alg_number Algorithm number.
 * @param[out] outputdir Output directory path.
 * @return 0 on success, -1 on error.
 */
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
 * @brief Generates a random array based on the specified type and size.
 *
 * @param target Pointer to the array to fill with random values.
 * @param type_string Data type as a string.
 * @param array_size Number of elements in the array.
 * @param rank MPI rank to seed the random number generator.
 * @return 0 on success, -1 if the data type is unsupported.
 */
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


/**
 * @brief Concatenates a directory path and a filename into a full file path.
 *
 * @param dir_path Directory path.
 * @param filename Filename to append.
 * @param fullpath Buffer where the concatenated path is stored.
 * @return 0 on success, -1 on error.
 */
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


/**
 * @brief Writes the timing results to a specified output file.
 *
 * This function writes the highest timing values and the timings for each rank 
 * to a file in CSV format. The header includes "highest" followed by each rank 
 * from rank0 to rankN (where N is the number of ranks).
 *
 * @param fullpath The full path to the output file.
 * @param highest An array containing the highest timing values for each iteration.
 * @param all_times A 2D array flattened into 1D containing timing values for all ranks 
 *                  across all iterations.
 * @param iter The number of iterations.
 *
 * @return int Returns 0 on success, or -1 if an error occurs while opening the file.
 *
 * @note Time is saved in ns (i.e. 10^-9 s).
 */
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

/**
 * @brief Checks if a file does not exists.
 *
 * @param filename The name of the file to check.
 * @return int Returns 1 if the file does not exists, 0 otherwise.
 */
int file_not_exists(const char* filename) {
  struct stat buffer;
  return (stat(filename, &buffer) != 0) ? 1 : 0;
}


/**
 * @brief Writes MPI rank and processor name allocations to a specified file using MPI I/O.
 *
 * This function collects the MPI rank and processor name for each process and writes
 * them to a file in CSV format using parallel I/O. The file will have the format:
 *
 * rank,allocation
 * 0,processor_name_0
 * 1,processor_name_1
 * ...
 *
 * Each rank calculates its unique offset using a fixed entry size, based on MPI_MAX_PROCESSOR_NAME,
 * ensuring non-overlapping writes without requiring data size communication or gathering.
 *
 * This implementation uses `MPI_File_write_at` for concurrent, safe file access and a barrier
 * to synchronize ranks after writing the header.
 *
 * @param filename The name of the file to which the allocations will be written.
 * @param comm The MPI communicator.
 *
 * @return int Returns MPI_SUCCESS on success, MPI_ERR otherwise.
 */
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


