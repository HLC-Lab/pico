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
  COLL_UNKNOWN
}rules_coll_t;


rules_coll_t get_collective_from_string(const char *coll_str) {
  if (strcmp(coll_str, "ALLREDUCE") == 0)       return ALLREDUCE;
  if (strcmp(coll_str, "ALLGATHER") == 0)       return ALLGATHER;
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
      if (new_value < 0 || new_value > 16) {
        fprintf(stderr, "ERROR: For allreduce, the number must be between 0 and 16.\n");
        return -1;
      }
      #else
      if (new_value < 0 || new_value > 16 || (new_value > 7 && new_value < 14)) {
        fprintf(stderr, "ERROR: For allreduce, the number must be between 0 and 7 or between 14 and 16.\n");
        return -1;
      }
      #endif
      if (new_value > 13) new_value = 0;
      target_line = 12;
      break;
    case ALLGATHER:
      if (new_value < 0 || new_value > 10) {
        fprintf(stderr, "ERROR: For allgather, the number must be between 0 and 10.\n");
        return -1;
      }
      if (new_value > 7) new_value = 0;
      target_line = 6;
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
