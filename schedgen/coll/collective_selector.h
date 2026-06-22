/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */
#ifndef SCHEDGEN_COLL_COLLECTIVE_SELECTOR_H
#define SCHEDGEN_COLL_COLLECTIVE_SELECTOR_H

#include <map>
#include <string>
#include <vector>

#include "collective_registry.h"

struct CollectiveRule {
  int min_comm = 0;
  int max_comm = -1;
  int min_msg_size = 0;
  int max_msg_size = -1;
  std::string algorithm;
};

class CollectiveSelector {
public:
  void load_default_rules();
  bool load_rules_file(const std::string &path);
  std::string choose(CollectiveKind kind, int comm_size,
                     int msg_size) const;
  std::string fallback_algorithm(CollectiveKind kind) const;

private:
  struct RuleSet {
    std::string fallback_algorithm;
    std::vector<CollectiveRule> rules;
  };

  std::map<CollectiveKind, RuleSet> rule_map;

  static bool match_rule(const CollectiveRule &rule, int comm_size,
                         int msg_size);
};

#endif // SCHEDGEN_COLL_COLLECTIVE_SELECTOR_H
