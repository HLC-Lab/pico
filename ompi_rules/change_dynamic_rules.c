#include <stdio.h>
#include <stdlib.h>

#include "change_dynamic_rules_utils.h"


int main(int argc, char *argv[]) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <filename> <algorithm number>\n", argv[0]);
    return EXIT_FAILURE;
  }

  const char *filename = argv[1];
  int new_value = atoi(argv[2]);
  const char *coll_str = NULL;

  coll_str = getenv("COLLECTIVE_TYPE");
  if (NULL == coll_str) {
    fprintf(stderr, "Error! `COLLECTIVE_TYPE` environment variable not set. Aborting...\n");
    return EXIT_FAILURE;
  }

  if (update_file(filename, new_value, get_collective_from_string(coll_str)) != 0){
    return EXIT_FAILURE;
 }

  return EXIT_SUCCESS;
}
