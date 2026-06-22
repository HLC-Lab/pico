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

#include "schedgen_coll_helper.h"

void create_gather(gengetopt_args_info *args_info) {

  /**
   * In this pattern every rank (except root) sends data to the root, which has
   * to receive it. Altough in MPI_Gather root sends data to himself, we are not
   * doing it that way here, because one can assume that a good MPI
   * imnplementation would just copy the data locally.
   */

  int comm_size = args_info->commsize_arg;
  int datasize = args_info->datasize_arg;

  Goal goal(args_info, comm_size);

  for (int comm_rank = 0; comm_rank < comm_size; comm_rank++) {
    goal.StartRank(comm_rank);
    int vrank = rank_to_vrank(comm_rank, args_info->root_arg);

    if (vrank == 0) {
      for (int i = 1; i < comm_size; i++) {
        int recvpeer = vrank_to_rank(i, args_info->root_arg);
        goal.Recv(datasize, recvpeer);
      }
    } else {
      int sendpeer = vrank_to_rank(0, args_info->root_arg);
      goal.Send(datasize, sendpeer);
    }
    goal.EndRank();
  }
  goal.Write();
}
