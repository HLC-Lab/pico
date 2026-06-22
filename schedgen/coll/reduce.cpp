/*
 * Copyright (c) 2009 The Trustees of Indiana University and Indiana
 *                    University Research and Technology
 *                    Corporation.  All rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *            Timo Schneider <timoschn@cs.indiana.edu>
 *
 */

#include "../schedgen.hpp"

#include <cmath>

#include "collective_registry.h"
#include "schedgen_coll_helper.h"

void create_binomial_tree_reduce_rank(Goal *goal, int root, int comm_rank,
                                      int comm_size, int datasize) {
  int vrank = rank_to_vrank(comm_rank, root);

  Goal::t_id send = -1;
  Goal::t_id recv = -1;
  int max_steps = static_cast<int>(std::ceil(std::log2(static_cast<double>(comm_size))));
  for (int r = 0; r < max_steps; r++) {
    int offset = 1 << r;
    int vpeer = vrank + offset;
    int peer = vrank_to_rank(vpeer, root);
    if ((vrank + offset < comm_size) && (vrank < offset)) {
      recv = goal->Recv(datasize, peer);
    }
    if ((send >= 0) && (recv >= 0)) {
      goal->Requires(send, recv);
    }
    vpeer = vrank - offset;
    peer = vrank_to_rank(vpeer, root);
    if ((vrank >= offset) && (vrank < (1 << (r + 1)))) {
      send = goal->Send(datasize, peer);
    }
  }
}

void create_binomial_tree_reduce(gengetopt_args_info *args_info) {

  int comm_size = args_info->commsize_arg;
  int datasize = args_info->datasize_arg;

  Goal goal(args_info, comm_size);

  for (int comm_rank = 0; comm_rank < comm_size; comm_rank++) {
    goal.StartRank(comm_rank);
    create_binomial_tree_reduce_rank(&goal, args_info->root_arg, comm_rank,
                                     comm_size, datasize);
    goal.EndRank();
  }
  goal.Write();
}

namespace {

void binomial_reduce(Goal *goal, int rank, int comm_size, int datasize,
                     const CollectiveContext &ctx) {
  create_binomial_tree_reduce_rank(goal, ctx.root, rank, comm_size, datasize);
}

bool register_algorithms() {
  register_collective_algorithm(CollectiveKind::Reduce, "binomial",
                                binomial_reduce);
  return true;
}

const bool registered = register_algorithms();

} // namespace
