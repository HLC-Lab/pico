#ifndef CHANGE_DYNAMIC_RULES_UTILS_H
#define CHANGE_DYNAMIC_RULES_UTILS_H

#include <stdio.h>
#include <string.h>

#define RULES_MAX_LINE_LENGTH 256

/**
 * @enum rules_coll_t
 *
 * @brief Defines the collective operation to be used in this test. It only provide symbolic
 * name for collective selection.
 * */
typedef enum{
  ALLREDUCE = 0,
  ALLGATHER,
  REDUCE_SCATTER,
  COLL_UNKNOWN
}rules_coll_t;


rules_coll_t get_collective_from_string(const char *coll_str) {
  if (strcmp(coll_str, "ALLREDUCE") == 0)       return ALLREDUCE;
  if (strcmp(coll_str, "ALLGATHER") == 0)       return ALLGATHER;
  if (strcmp(coll_str, "REDUCE_SCATTER") == 0)  return REDUCE_SCATTER;
  return COLL_UNKNOWN;
}

int update_file(const char *filename, int new_value, rules_coll_t coll) {
  // Used to indicate the line to be updated to change the dynamic
  // rule associated with the collective operation in environment variable
  // WARNING: This is a hardcoded value and must be updated if the rules file changes
  int target_line = 0;

  switch (coll) {
    case ALLREDUCE:
      #ifdef OMPI_TEST
      if (!((new_value >= 0 && new_value <= 7) ||       // [0-7] (default ompi)
            (new_value >= 8 && new_value <= 13) ||      // [8-13] (ompi test)
            (new_value >= 101 && new_value <= 103) ||   // [101-103] (default over)
            (new_value >= 201 && new_value <= 202))) {  // [201-202] (swing over)
        fprintf(stderr, "ERROR: Algorithm not in [0-7](default ompi) [8-13](ompi test) [101-103](default over) [201-202](swing over).\n");
        return -1;
      }
      #else
      if (!((new_value >= 0 && new_value <= 7) ||       // [0-7] (default ompi)
            (new_value >= 101 && new_value <= 103) ||   // [101-103] (default over)
            (new_value >= 201 && new_value <= 202))) {  // [201-202] (swing over)
        fprintf(stderr, "ERROR: ALLREDUCE algorithm not in [0-7](default ompi) [101-103](default over) [201-202](swing over).\n");
        return -1;
      }
      #endif
      if (new_value > 13) new_value = 0;
      target_line = 12;
      break;
    case ALLGATHER:
      if (!((new_value >= 0 && new_value <= 6) ||       // [0-6] (default ompi)
            (new_value >= 101 && new_value <= 103) ||   // [101-103] (default over)
            (new_value == 201))) {                      // [201] (swing over)
        fprintf(stderr, "ERROR: ALLGATHER algorithm not in [0-6](default ompi) [101-103](default over) [201](swing over).\n");
        return -1;
      }
      if (new_value > 6) new_value = 0;
      target_line = 6;
      break;
    case REDUCE_SCATTER:
      if (!((new_value >= 0 && new_value <= 4) ||       // [0-4] (default ompi)
            (new_value >= 101 && new_value <= 103))) {  // [101-103] (default over)
        fprintf(stderr, "ERROR: REDUCE_SCATTER algorithm not in [0-4](default ompi) [101-103](default over).\n");
        return -1;
      }
      if (new_value > 4) new_value = 0;
      target_line = 18;
      break;
    default:
      fprintf(stderr, "Error: Invalid collective type.\n");
      return -1;
  }

  FILE *file = fopen(filename, "r+");
  if (file == NULL) {
    fprintf(stderr, "Error opening the file.\n");
    return -1;
  }

  char line[RULES_MAX_LINE_LENGTH];
  int line_number = 0;
  long line_position = 0;

  // Find the position of the target_line line
  while (fgets(line, sizeof(line), file)) {
    line_number++;
    if (line_number == target_line) {
      // Save the position of the target_line
      line_position = ftell(file) - strlen(line);
      break;
    }
  }

  if (line_number != target_line) {
    fprintf(stderr, "Error: File does not have enough lines.\n");
    fclose(file);
    return -1;
  }

  // Move back to the start of the sixth line and overwrite it
  fseek(file, line_position, SEEK_SET);
  fprintf(file, "0 %d 0 0 # Algorithm\n", new_value);

  fclose(file);
  return 0;
}
#endif
