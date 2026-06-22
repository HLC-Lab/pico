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
#include <vector>

#include "collective_registry.h"
#include "schedgen_coll_helper.h"

void create_dissemination_rank(Goal *goal, int comm_rank, int comm_size,
                               int datasize) {
  Goal::t_id req = Goal::NO_ID;
  for (int round = 0; round < std::ceil(std::log2(static_cast<double>(comm_size))); round++) {
    int send_to = mymod(comm_rank + (1 << round), comm_size);
    int recv_from = mymod(comm_rank - (1 << round), comm_size);
    int send = goal->Send(datasize, send_to);
    if (req != Goal::NO_ID) {
      goal->Requires(send, req);
    }
    req = goal->Recv(datasize, recv_from);
  }
}

void create_dissemination(gengetopt_args_info *args_info) {
  int comm_size = args_info->commsize_arg;
  int datasize = args_info->datasize_arg;

  Goal goal(args_info, comm_size);

  for (int comm_rank = 0; comm_rank < comm_size; comm_rank++) {
    goal.StartRank(comm_rank);
    create_dissemination_rank(&goal, comm_rank, comm_size, datasize);
    goal.EndRank();
  }
  goal.Write();
}

void create_linbarrier(gengetopt_args_info *args_info) {

  int comm_size = args_info->commsize_arg;
  Goal goal(args_info, comm_size);

  for (int comm_rank = 0; comm_rank < comm_size; comm_rank++) {
    goal.StartRank(comm_rank);

    if (comm_rank == 0) {
      int dummy = goal.Exec("redfunc", 0);
      for (int i = 1; i < comm_size; i++) {
        int recv = goal.Recv(1, i);
        int send = goal.Send(1, i);
        goal.Requires(dummy, recv);
        goal.Requires(send, dummy);
      }
    } else {
      goal.Send(1, 0);
      goal.Recv(1, 0);
    }

    goal.EndRank();
  }
  goal.Write();
}

void create_nway_dissemination(gengetopt_args_info *args_info) {

  int n = args_info->nway_arg;
  int comm_size = args_info->commsize_arg;
  int datasize = args_info->datasize_arg;

  Goal goal(args_info, comm_size);

  for (int comm_rank = 0; comm_rank < comm_size; comm_rank++) {

    goal.StartRank(comm_rank);

    std::vector<int> recv(n, -1);
    std::vector<int> send(n, -1);

    int rounds = static_cast<int>(std::ceil(mylog(n + 1, comm_size)));
    for (int r = 0; r < rounds; r++) {
      for (int w = 1; w <= n; w++) {
        int sendpeer =
            mymod(comm_rank + w * static_cast<int>(std::pow(n + 1, r)), comm_size);
        send[w - 1] = goal.Send(datasize, sendpeer);
      }
      if (r > 0) {
        int prev = recv[0];
        for (int w = 1; w < n; w++) {
          int red = goal.Exec("redfunc", datasize);
          goal.Requires(red, recv[w]);
          goal.Requires(red, prev);
          prev = red;
        }
        int red = goal.Exec("redfunc", datasize);
        goal.Requires(red, recv[0]);
        goal.Requires(send[n - 1], prev);
        for (int w = 1; w <= n; w++) {
          goal.Requires(send[w - 1], red);
        }
      }
      for (int w = 1; w <= n; w++) {
        int recvpeer =
            mymod(comm_rank - w * static_cast<int>(std::pow(n + 1, r)), comm_size);
        recv[w - 1] = goal.Recv(datasize, recvpeer);
      }
    }
    goal.EndRank();
  }
  goal.Write();
}

namespace {

void dissemination_barrier(Goal *goal, int rank, int comm_size, int datasize,
                           const CollectiveContext &ctx) {
  (void)ctx;
  create_dissemination_rank(goal, rank, comm_size, datasize);
}

bool register_algorithms() {
  register_collective_algorithm(CollectiveKind::Barrier, "dissemination",
                                dissemination_barrier);
  register_collective_algorithm(CollectiveKind::Iallreduce, "dissemination",
                                dissemination_barrier);
  register_collective_algorithm(CollectiveKind::Allgather, "dissemination",
                                dissemination_barrier);
  return true;
}

const bool registered = register_algorithms();

} // namespace
