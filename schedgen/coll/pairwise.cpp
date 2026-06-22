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

#include <cassert>
#include <vector>

#include "../MersenneTwister.h"
#include "schedgen_coll_helper.h"

void create_double_ring(gengetopt_args_info *args_info) {

  int comm_size = args_info->commsize_arg;
  int datasize = args_info->datasize_arg;

  Goal goal(args_info, comm_size);

  for (int comm_rank = 0; comm_rank < comm_size; comm_rank++) {
    goal.StartRank(comm_rank);

    /* forward ring */
    int send = goal.Send(datasize, ((comm_rank + 1) + comm_size) % comm_size);
    int recv = goal.Recv(datasize, ((comm_rank - 1) + comm_size) % comm_size);
    if (comm_rank > 0)
      goal.Requires(send, recv);

    /* backward ring */
    send = goal.Send(datasize, ((comm_rank - 1) + comm_size) % comm_size);
    recv = goal.Recv(datasize, ((comm_rank + 1) + comm_size) % comm_size);
    if (comm_rank > 0)
      goal.Requires(send, recv);

    goal.EndRank();
  }
  goal.Write();
}

void create_random_bisect(gengetopt_args_info *args_info) {

  int comm_size = args_info->commsize_arg;
  if (comm_size % 2 == 1)
    comm_size--;
  int datasize = args_info->datasize_arg;
  Goal goal(args_info, comm_size);

  MTRand mtrand;
  std::vector<int> peer(
      comm_size); // save the pairs (peer[i] is the peer of host i)
  std::vector<bool> used(comm_size, false); // mark the used peers

  // quick method to create a random pairing
  for (int counter = 0; counter < comm_size; counter++) {
    int myrand = mtrand.randInt(comm_size - counter - 1);
    int pos = 0;
    while (true) {
      // walk the used array (only the entries that are not used)
      if (used[pos] == false) {
        if (myrand == 0) {
          used[pos] = true;
          peer[counter] = pos; // save random value
          break;
        }
        myrand--;
      }
      pos++;
      assert(pos < comm_size);
    }
  }

  // create the inverse array ...
  std::vector<int> inverse_peer(
      comm_size); // the inverse peer table (know who to receive from)
  for (int comm_rank = 0; comm_rank < comm_size; comm_rank++) {
    inverse_peer[peer[comm_rank]] = comm_rank;
  }

  for (int comm_rank = 0; comm_rank < comm_size; comm_rank++) {
    goal.StartRank(comm_rank);

    int dist = comm_size / 2;

    if (inverse_peer[comm_rank] < dist) {
      // this host is a sender
      goal.Send(datasize, peer[inverse_peer[comm_rank] + dist]);
    } else {
      // this host is a receiver
      goal.Recv(datasize, peer[inverse_peer[comm_rank] - dist]);
    }

    goal.EndRank();
  }
  goal.Write();
}

void create_random_bisect_fd_sym(gengetopt_args_info *args_info) {

  int comm_size = args_info->commsize_arg;
  if (comm_size % 2 == 1)
    comm_size--;
  int datasize = args_info->datasize_arg;
  Goal goal(args_info, comm_size);

  MTRand mtrand;
  std::vector<int> peer(
      comm_size); // save the pairs (peer[i] is the peer of host i)
  std::vector<bool> used(comm_size, false); // mark the used peers

  // quick method to create a random pairing
  for (int counter = 0; counter < comm_size; counter++) {
    int myrand = mtrand.randInt(comm_size - counter - 1);
    int pos = 0;
    while (true) {
      // walk the used array (only the entries that are not used)
      if (used[pos] == false) {
        if (myrand == 0) {
          used[pos] = true;
          peer[counter] = pos; // save random value
          break;
        }
        myrand--;
      }
      pos++;
      assert(pos < comm_size);
    }
  }

  // create the inverse array ...
  std::vector<int> inverse_peer(
      comm_size); // the inverse peer table (know who to receive from)
  for (int comm_rank = 0; comm_rank < comm_size; comm_rank++) {
    inverse_peer[peer[comm_rank]] = comm_rank;
  }

  for (int comm_rank = 0; comm_rank < comm_size; comm_rank++) {
    goal.StartRank(comm_rank);

    int dist = comm_size / 2;

    if (inverse_peer[comm_rank] < dist) {
      // this host is a sender
      goal.Send(datasize, peer[inverse_peer[comm_rank] + dist]);
      goal.Recv(datasize, peer[inverse_peer[comm_rank] + dist]);
    } else {
      // this host is a receiver
      goal.Recv(datasize, peer[inverse_peer[comm_rank] - dist]);
      goal.Send(datasize, peer[inverse_peer[comm_rank] - dist]);
    }

    goal.EndRank();
  }
  goal.Write();
}
