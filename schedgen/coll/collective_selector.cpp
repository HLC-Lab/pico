/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */
#include "collective_selector.h"

#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {

constexpr int kUnbounded = std::numeric_limits<int>::max();

CollectiveKind parse_collective_kind(const std::string &name) {
  if (name == "barrier")
    return CollectiveKind::Barrier;
  if (name == "allreduce")
    return CollectiveKind::Allreduce;
  if (name == "iallreduce")
    return CollectiveKind::Iallreduce;
  if (name == "bcast")
    return CollectiveKind::Bcast;
  if (name == "allgather")
    return CollectiveKind::Allgather;
  if (name == "reduce")
    return CollectiveKind::Reduce;
  if (name == "alltoall")
    return CollectiveKind::Alltoall;
  throw std::runtime_error("Unknown collective kind: " + name);
}

int parse_bound(const std::string &token, bool upper) {
  if (token == "*" || token.empty()) {
    return upper ? kUnbounded : 0;
  }
  return std::stoi(token);
}

} // namespace

void CollectiveSelector::load_default_rules() {
  rule_map.clear();
  auto set_rule = [&](CollectiveKind kind, const std::string &fallback) {
    rule_map[kind].fallback_algorithm = fallback;
  };

  set_rule(CollectiveKind::Barrier, "dissemination");
  set_rule(CollectiveKind::Allreduce, "recursive_doubling");
  set_rule(CollectiveKind::Iallreduce, "dissemination");
  set_rule(CollectiveKind::Bcast, "binomial");
  set_rule(CollectiveKind::Allgather, "dissemination");
  set_rule(CollectiveKind::Reduce, "binomial");
  set_rule(CollectiveKind::Alltoall, "linear");
}

bool CollectiveSelector::load_rules_file(const std::string &path) {
  std::ifstream in(path.c_str());
  if (!in.is_open()) {
    return false;
  }

  std::string line;
  int lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream iss(line);
    std::string coll_name, algorithm, min_comm_s, max_comm_s, min_msg_s,
        max_msg_s;
    if (!(iss >> coll_name >> algorithm >> min_comm_s >> max_comm_s >>
          min_msg_s >> max_msg_s)) {
      throw std::runtime_error("Invalid rule line " + std::to_string(lineno) +
                               " in " + path);
    }
    CollectiveKind kind = parse_collective_kind(coll_name);
    CollectiveRule rule;
    rule.algorithm = algorithm;
    rule.min_comm = parse_bound(min_comm_s, false);
    rule.max_comm = parse_bound(max_comm_s, true);
    rule.min_msg_size = parse_bound(min_msg_s, false);
    rule.max_msg_size = parse_bound(max_msg_s, true);
    rule_map[kind].rules.push_back(rule);
  }
  return true;
}

std::string CollectiveSelector::choose(CollectiveKind kind, int comm_size,
                                       int msg_size) const {
  auto it = rule_map.find(kind);
  if (it == rule_map.end()) {
    return std::string();
  }
  const RuleSet &set = it->second;
  for (const auto &rule : set.rules) {
    if (match_rule(rule, comm_size, msg_size)) {
      return rule.algorithm;
    }
  }
  return set.fallback_algorithm;
}

std::string CollectiveSelector::fallback_algorithm(CollectiveKind kind) const {
  auto it = rule_map.find(kind);
  if (it == rule_map.end()) {
    return std::string();
  }
  return it->second.fallback_algorithm;
}

bool CollectiveSelector::match_rule(const CollectiveRule &rule, int comm_size,
                                    int msg_size) {
  if (comm_size < rule.min_comm)
    return false;
  if (rule.max_comm != -1 && comm_size > rule.max_comm)
    return false;
  if (msg_size < rule.min_msg_size)
    return false;
  if (rule.max_msg_size != -1 && msg_size > rule.max_msg_size)
    return false;
  return true;
}
