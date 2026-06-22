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

#include <cstdlib>
#include <vector>

#include "collective_registry.h"
#include "schedgen_coll_helper.h"

void create_linear_alltoall_rank(Goal *goal, int src_rank, int comm_size,
                                 int datasize) {
  for (int step = 1; step < comm_size; step++) {
    int send_to = (src_rank + step) % comm_size;
    int recv_from = mymod(src_rank - step, comm_size);
    goal->Send(datasize, send_to);
    goal->Recv(datasize, recv_from);
  }
}

void create_linear_alltoall(gengetopt_args_info *args_info) {
  int comm_size = args_info->commsize_arg;
  int datasize = args_info->datasize_arg;

  Goal goal(args_info, comm_size);

  for (int src_rank = 0; src_rank < comm_size; src_rank++) {
    goal.StartRank(src_rank);
    create_linear_alltoall_rank(&goal, src_rank, comm_size, datasize);
    goal.EndRank();
  }
  goal.Write();
}

void create_linear_alltoallv_rank(Goal *goal, int src_rank, int comm_size,
                                  std::vector<std::vector<int>> &sizes) {
  for (int step = 1; step < comm_size; step++) {
    int send_to = (src_rank + step) % comm_size;
    int recv_from = mymod(src_rank - step, comm_size);
    goal->Send(sizes[src_rank][send_to], send_to);
    goal->Recv(sizes[recv_from][src_rank], recv_from);
  }
}

void create_linear_alltoallv(gengetopt_args_info *args_info) {
  int comm_size = args_info->commsize_arg;
  int datasize = args_info->datasize_arg;

  Goal goal(args_info, comm_size);
  std::vector<std::vector<int>> sizes(comm_size);
  for (int i = 0; i < comm_size; i++) {
    sizes[i].reserve(comm_size);
    for (int j = 0; j < comm_size; j++) {
      if (j == args_info->root_arg) {
        sizes[i].push_back(datasize);
      } else if (args_info->outcast_flag &&
                 i == (args_info->root_arg + 1) % comm_size) {
        sizes[i].push_back(datasize);
      } else {
        sizes[i].push_back(
            (rand() % (datasize / args_info->a2av_skew_ratio_arg)) + 1);
      }
    }
  }

  for (int src_rank = 0; src_rank < comm_size; src_rank++) {
    goal.StartRank(src_rank);
    create_linear_alltoallv_rank(&goal, src_rank, comm_size, sizes);
    goal.EndRank();
  }
  goal.Write();
}

namespace {

void linear_alltoall(Goal *goal, int rank, int comm_size, int datasize,
                     const CollectiveContext &ctx) {
  (void)ctx;
  create_linear_alltoall_rank(goal, rank, comm_size, datasize);
}

bool register_algorithms() {
  register_collective_algorithm(CollectiveKind::Alltoall, "linear",
                                linear_alltoall);
  return true;
}

const bool registered = register_algorithms();

} // namespace
