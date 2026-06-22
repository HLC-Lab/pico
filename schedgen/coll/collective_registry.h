/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */
#ifndef SCHEDGEN_COLL_COLLECTIVE_REGISTRY_H
#define SCHEDGEN_COLL_COLLECTIVE_REGISTRY_H

#include <functional>
#include <map>
#include <string>
#include <vector>

class Goal;
struct CollectiveContext;

enum class CollectiveKind {
  Barrier,
  Allreduce,
  Iallreduce,
  Bcast,
  Allgather,
  Reduce,
  Alltoall
};

struct CollectiveContext {
  int root = 0;
  int replace_comp_time = -1;
  int segmentsize = 0;
  int element_count = 0;
  int datatype_bytes = 0;
};

using CollectiveGenerator =
    std::function<void(Goal *, int rank, int comm_size, int data_size,
                       const CollectiveContext &ctx)>;

using CollectiveValidator = std::function<bool(
    int comm_size, int data_size, const CollectiveContext &ctx,
    std::string *reason)>;

struct CollectiveAlgorithm {
  CollectiveGenerator generator;
  CollectiveValidator validator;

  explicit operator bool() const { return static_cast<bool>(generator); }
};

void register_collective_algorithm(CollectiveKind kind,
                                   const std::string &name,
                                   CollectiveGenerator generator,
                                   CollectiveValidator validator =
                                       CollectiveValidator());

CollectiveAlgorithm lookup_collective_algorithm(CollectiveKind kind,
                                                const std::string &name);

std::vector<std::string> list_collective_algorithms(CollectiveKind kind);

const char *collective_kind_to_string(CollectiveKind kind);

#endif // SCHEDGEN_COLL_COLLECTIVE_REGISTRY_H
