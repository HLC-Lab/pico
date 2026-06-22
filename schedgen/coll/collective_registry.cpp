/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */
#include "collective_registry.h"

#include <mutex>

namespace {

using AlgorithmMap = std::map<std::string, CollectiveAlgorithm>;

struct Registry {
  std::map<CollectiveKind, AlgorithmMap> per_collective;
  std::mutex mutex;
};

Registry &registry() {
  static Registry reg;
  return reg;
}

} // namespace

void register_collective_algorithm(CollectiveKind kind,
                                   const std::string &name,
                                   CollectiveGenerator generator,
                                   CollectiveValidator validator) {
  Registry &reg = registry();
  std::lock_guard<std::mutex> guard(reg.mutex);
  reg.per_collective[kind][name] =
      CollectiveAlgorithm{std::move(generator), std::move(validator)};
}

CollectiveAlgorithm lookup_collective_algorithm(CollectiveKind kind,
                                                const std::string &name) {
  Registry &reg = registry();
  std::lock_guard<std::mutex> guard(reg.mutex);
  auto coll_it = reg.per_collective.find(kind);
  if (coll_it == reg.per_collective.end()) {
    return CollectiveAlgorithm();
  }
  auto algo_it = coll_it->second.find(name);
  if (algo_it == coll_it->second.end()) {
    return CollectiveAlgorithm();
  }
  return algo_it->second;
}

std::vector<std::string> list_collective_algorithms(CollectiveKind kind) {
  Registry &reg = registry();
  std::lock_guard<std::mutex> guard(reg.mutex);
  std::vector<std::string> names;
  auto coll_it = reg.per_collective.find(kind);
  if (coll_it == reg.per_collective.end()) {
    return names;
  }
  for (const auto &entry : coll_it->second) {
    names.push_back(entry.first);
  }
  return names;
}

const char *collective_kind_to_string(CollectiveKind kind) {
  switch (kind) {
  case CollectiveKind::Barrier:
    return "barrier";
  case CollectiveKind::Allreduce:
    return "allreduce";
  case CollectiveKind::Iallreduce:
    return "iallreduce";
  case CollectiveKind::Bcast:
    return "bcast";
  case CollectiveKind::Allgather:
    return "allgather";
  case CollectiveKind::Reduce:
    return "reduce";
  case CollectiveKind::Alltoall:
    return "alltoall";
  }
  return "unknown";
}
