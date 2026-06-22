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
#include <queue>
#include <vector>

#include "collective_registry.h"
#include "schedgen_coll_helper.h"

void create_binomial_tree_bcast_rank(Goal *goal, int root, int comm_rank,
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
    int upper_bound = 1 << (r + 1);
    if ((vrank >= offset) && (vrank < upper_bound)) {
      send = goal->Send(datasize, peer);
    }
  }
}

void create_binomial_tree_bcast(gengetopt_args_info *args_info) {

  int comm_size = args_info->commsize_arg;
  int datasize = args_info->datasize_arg;

  Goal goal(args_info, comm_size);

  for (int comm_rank = 0; comm_rank < comm_size; comm_rank++) {
    goal.StartRank(comm_rank);
    create_binomial_tree_bcast_rank(&goal, args_info->root_arg, comm_rank,
                                    comm_size, datasize);
    goal.EndRank();
  }
  goal.Write();
}

void create_binary_tree_bcast_rank(Goal *goal, int root, int comm_rank,
                                   int comm_size, int datasize) {

  int vrank = rank_to_vrank(comm_rank, root);

  std::queue<int> q;

  q.push(vrank);

  int steps = 0;
  Goal::t_id red_step = Goal::NO_ID;
  while (!q.empty()) {

    std::queue<int> new_q;

    while (!q.empty()) {

      int src = q.front();
      q.pop();

      int tgt = 0;

      if (2 * src + 1 < comm_size) {

        if (red_step == Goal::NO_ID) {
          red_step = goal->Exec("redfunc", steps * 100, 0);
        }

        tgt = 2 * src + 1;

        int vtgt = tgt;
        int vsrc = src;
        vtgt = vrank_to_rank(vtgt, root);
        vsrc = vrank_to_rank(vsrc, root);

        if (comm_rank == vsrc) {
          Goal::t_id send = goal->Send(datasize, vtgt);
          goal->Requires(send, red_step);
        }

        if (comm_rank == vtgt) {
          Goal::t_id recv = goal->Recv(datasize, vsrc);
          goal->Requires(red_step, recv);
        }

        new_q.push(tgt);
      }

      if (2 * src + 2 < comm_size) {

        if (red_step == Goal::NO_ID) {
          red_step = goal->Exec("redfunc", steps * 100, 0);
        }

        tgt = 2 * src + 2;

        int vtgt = tgt;
        int vsrc = src;
        vtgt = vrank_to_rank(vtgt, root);
        vsrc = vrank_to_rank(vsrc, root);

        if (comm_rank == vsrc) {
          Goal::t_id send = goal->Send(datasize, vtgt);
          goal->Requires(send, red_step);
        }

        if (comm_rank == vtgt) {
          Goal::t_id recv = goal->Recv(datasize, vsrc);
          goal->Requires(red_step, recv);
        }

        new_q.push(tgt);
      }
    }
    steps++;
    q.swap(new_q);
  }
}

void create_binary_tree_bcast(gengetopt_args_info *args_info) {

  int comm_size = args_info->commsize_arg;
  int datasize = args_info->datasize_arg;

  Goal goal(args_info, comm_size);

  for (int comm_rank = 0; comm_rank < comm_size; comm_rank++) {
    goal.StartRank(comm_rank);
    create_binary_tree_bcast_rank(&goal, args_info->root_arg, comm_rank,
                                  comm_size, datasize);
    goal.EndRank();
  }
  goal.Write();
}

void create_pipelined_ring(gengetopt_args_info *args_info) {

  int comm_size = args_info->commsize_arg;
  int datasize = args_info->datasize_arg;
  int segmentsize = args_info->segmentsize_arg;

  Goal goal(args_info, comm_size);

  for (int comm_rank = 0; comm_rank < comm_size; comm_rank++) {
    goal.StartRank(comm_rank);
    int vrank = rank_to_vrank(comm_rank, args_info->root_arg);

    int num_rounds =
        static_cast<int>(std::ceil(static_cast<double>(datasize) /
                                   static_cast<double>(segmentsize)));

    std::vector<int> send(num_rounds, -1);
    std::vector<int> recv(num_rounds, -1);

    for (int r = 0; r < num_rounds; r++) {

      int recvpeer = vrank - 1;
      int sendpeer = vrank + 1;

      int psize = segmentsize;
      if (r == num_rounds - 1) {
        psize = datasize - segmentsize * r;
      }

      int vpeer = recvpeer;
      recvpeer = vrank_to_rank(vpeer, args_info->root_arg);
      if (recvpeer >= 0) {
        recv.at(r) = goal.Recv(psize, recvpeer);
      }

      vpeer = sendpeer;
      sendpeer = vrank_to_rank(vpeer, args_info->root_arg);
      if (sendpeer < comm_size) {
        send.at(r) = goal.Send(psize, sendpeer);
      }

      if ((send.at(r) > 0) && (recv.at(r) > 0)) {
        goal.Requires(send.at(r), recv.at(r));
      }
    }
    goal.EndRank();
    if (comm_rank == comm_size - 1) {
      goal.Write();
    }
  }
}

void create_pipelined_ring_dep(gengetopt_args_info *args_info) {

  int comm_size = args_info->commsize_arg;
  int datasize = args_info->datasize_arg;
  int segmentsize = args_info->segmentsize_arg;

  Goal goal(args_info, comm_size);

  for (int comm_rank = 0; comm_rank < comm_size; comm_rank++) {
    goal.StartRank(comm_rank);
    int vrank = rank_to_vrank(comm_rank, args_info->root_arg);

    int num_rounds = (int)ceil((double)datasize / (double)segmentsize);

    std::vector<int> send(num_rounds, -1);
    std::vector<int> recv(num_rounds, -1);

    for (int r = 0; r < num_rounds; r++) {

      int recvpeer = vrank - 1;
      int sendpeer = vrank + 1;

      // how much data is transmitted in this round
      int psize = segmentsize;
      if (r == num_rounds - 1) {
        psize = datasize - segmentsize * r;
      }

      // recv (if we are a receiver)
      int vpeer = recvpeer;
      recvpeer = vrank_to_rank(vpeer, args_info->root_arg);
      if (vpeer >= 0) {
        recv.at(r) = goal.Recv(psize, recvpeer);
      }

      // send (if we are a sender)
      vpeer = sendpeer;
      sendpeer = vrank_to_rank(vpeer, args_info->root_arg);
      if (vpeer < comm_size) {
        send.at(r) = goal.Send(psize, sendpeer);
      }

      // we can not send data before we received it
      if ((send.at(r) > 0) and (recv.at(r) > 0)) {
        goal.Requires(send.at(r), recv.at(r));
      }

      // ensure pipelining (only receive round r after r-1 send finished)
      if ((r > 0) and (send.at(r - 1) > 0) and (recv.at(r) > 0)) {
        goal.Requires(recv.at(r), send.at(r - 1));
      }
    }
    goal.EndRank();
    if (comm_rank == comm_size - 1)
      goal.Write();
  }
}

namespace {

void binomial_bcast(Goal *goal, int rank, int comm_size, int datasize,
                    const CollectiveContext &ctx) {
  create_binomial_tree_bcast_rank(goal, ctx.root, rank, comm_size, datasize);
}

void binary_bcast(Goal *goal, int rank, int comm_size, int datasize,
                  const CollectiveContext &ctx) {
  create_binary_tree_bcast_rank(goal, ctx.root, rank, comm_size, datasize);
}

bool register_algorithms() {
  register_collective_algorithm(CollectiveKind::Bcast, "binomial",
                                binomial_bcast);
  register_collective_algorithm(CollectiveKind::Bcast, "binary",
                                binary_bcast);
  return true;
}

const bool registered = register_algorithms();

} // namespace
