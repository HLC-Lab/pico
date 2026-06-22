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

#include <string>
#include <utility>
#include <vector>

#include "collective_registry.h"
#include "schedgen_coll_helper.h"

void create_allreduce_recdoub_rank(Goal *goal, int src_rank, int comm_size,
                                   int data_size) {
  int pof2 = comm_size_pof2(comm_size);
  int rem = comm_size - pof2;
  int new_rank = -1;
  int mask = 0x1;
  int new_dst = -1;
  int dst = -1;
  int prev_recv = -1;
  int send_id = -1;
  int recv_id = -1;

  if (src_rank < 2 * rem) {
    if (src_rank % 2 == 0) {
      send_id = goal->Send(data_size, src_rank + 1);
      new_rank = -1;
    } else {
      prev_recv = goal->Recv(data_size, src_rank - 1);
      new_rank = src_rank / 2;
    }
  } else {
    new_rank = src_rank - rem;
  }

  if (new_rank != -1) {
    while (mask < pof2) {
      new_dst = new_rank ^ mask;
      dst = (new_dst < rem) ? new_dst * 2 + 1 : new_dst + rem;
      send_id = goal->Send(data_size, dst);
      if (prev_recv != -1) {
        goal->Requires(send_id, prev_recv);
      }
      prev_recv = goal->Recv(data_size, dst);
      mask <<= 1;
    }
  }

  if (src_rank < 2 * rem) {
    if (src_rank % 2 == 1) {
      send_id = goal->Send(data_size, src_rank - 1);
      if (prev_recv != -1) {
        goal->Requires(send_id, prev_recv);
      }
    } else {
      recv_id = goal->Recv(data_size, src_rank + 1);
      (void)recv_id;
    }
  }
}

void create_reduce_scatter_allgather_rank(Goal *goal, int src_rank,
                                          int comm_size, int datasize,
                                          int replace_comptime) {
  int mask = 0x1;
  int next_datasize = datasize;
  int last_recv = -1;
  int send_id = -1;
  int num_steps_per_phase =
      static_cast<int>(std::log2(static_cast<double>(comm_size)));
  for (int step = 0; step < num_steps_per_phase; step++) {
    int dest = src_rank ^ mask;
    next_datasize /= 2;
    send_id = goal->Send(next_datasize, dest);
    if (last_recv != -1) {
      goal->Requires(send_id, last_recv);
    }
    last_recv = goal->Recv(next_datasize, dest);
    if (replace_comptime != -1) {
      last_recv = goal->Exec("intermsg-gap", replace_comptime, 0);
    }
    mask <<= 1;
  }

  mask >>= 1;
  for (int step = 0; step < num_steps_per_phase; step++) {
    int dest = src_rank ^ mask;
    send_id = goal->Send(next_datasize, dest);
    if (last_recv != -1) {
      goal->Requires(send_id, last_recv);
    }
    last_recv = goal->Recv(next_datasize, dest);
    if (replace_comptime != -1 && step != num_steps_per_phase - 1) {
      last_recv = goal->Exec("intermsg-gap", replace_comptime, 0);
    }
    next_datasize *= 2;
    mask >>= 1;
  }
}

void create_allreduce_recdoub(gengetopt_args_info *args_info) {
  int comm_size = args_info->commsize_arg;
  int datasize = args_info->datasize_arg;
  int replace_comptime = args_info->rpl_dep_cmp_arg;

  Goal goal(args_info, comm_size);

  for (int src_rank = 0; src_rank < comm_size; src_rank++) {
    goal.StartRank(src_rank);
    create_reduce_scatter_allgather_rank(&goal, src_rank, comm_size, datasize,
                                         replace_comptime);
    goal.EndRank();
  }
  goal.Write();
}

void create_allreduce_ring_rank(Goal *goal, int src_rank, int comm_size,
                                int datasize) {
  int last_recv = -1;
  int send_id = -1;
  int chunk_size = datasize;
  for (int phase = 0; phase < 2; phase++) {
    for (int step = 0; step < comm_size - 1; step++) {
      int send_to = (src_rank + 1) % comm_size;
      int recv_from = mymod(src_rank - 1, comm_size);
      send_id = goal->Send(chunk_size, send_to);
      if (last_recv != -1) {
        goal->Requires(send_id, last_recv);
      }
      last_recv = goal->Recv(chunk_size, recv_from);
    }
  }
}

void create_allreduce_ring(gengetopt_args_info *args_info) {
  int comm_size = args_info->commsize_arg;
  int datasize = args_info->datasize_arg;

  Goal goal(args_info, comm_size);

  for (int src_rank = 0; src_rank < comm_size; src_rank++) {
    goal.StartRank(src_rank);
    create_allreduce_ring_rank(&goal, src_rank, comm_size, datasize);
    goal.EndRank();
  }
  goal.Write();
}

void create_resnet152(gengetopt_args_info *args_info) {

  int collsbase = 100000; // needed to create tag, must be higher than the
                          // send/recvs in this schedule (0 here)
  int nops = 0;           // running count of colls for collective tag matching
  int comm = 1;           // only one comm used here
  int comm_size = args_info->commsize_arg;
  Goal goal(args_info, comm_size);

// The recipe for this was taken from https://github.com/spcl/DNN-cpp-proxies
// 1d32dce allreduce sizes for gradients with message aggregation
#define NUM_B 10
  int allreduce_sizes[NUM_B] = {6511592, 6567936, 5905920, 6113280, 6176256,
                                6112768, 6176256, 6112768, 5321216, 5194816};
  // batchsize = 128
  // Suggest world_size <= 256, which is corresponding to a global batch_size <=
  // 32 K A100 GPU runtime in us (10E-6) for each iteration
  int fwd_rt_whole_model = 119000;
  int bwd_rt_per_B = 23800;

  for (int src_rank = 0; src_rank < comm_size; src_rank++) {
    goal.StartRank(src_rank);
    int fwd_cmp = goal.Exec("forward_compute", fwd_rt_whole_model, 0); // compute
    for (int i = 0; i < NUM_B; i++) {
      // omitted progressing of MPI using Testany, no effect in goal
      int bkw_cmp = goal.Exec("backward_compute", bwd_rt_per_B, 0);
      goal.Requires(bkw_cmp, fwd_cmp);

      int dummy = goal.Exec("backward_compute_dummy", 0, 0);
      goal.Requires(dummy, bkw_cmp);

      // MPI_Iallreduce(allreduce_size[i], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD,
      // req[i]);
      goal.Comment("Iallreduce begin");
      goal.SetTag(MAKE_TAG(comm, (collsbase + nops)));
      goal.StartOp();
      create_dissemination_rank(&goal, src_rank, comm_size,
                                allreduce_sizes[i] * 4);
      std::pair<Goal::locop, Goal::locop> ops = goal.EndOp();
      Goal::locop::iterator it;
      for (it = ops.second.begin(); it != ops.second.end(); it++) {
        goal.Requires(it->first, dummy);
      }
      goal.Comment("Iallreduce end");
      nops++;
    }
    // MPI_Waitall(req);
    goal.EndRank();
  }
  goal.Write();
}

void create_chained_dissem(gengetopt_args_info *args_info) {

  int collsbase = 100000; // needed to create tag, must be higher than the
                          // send/recvs in this schedule (0 here)
  int comm = 1;           // only one comm used here
  int comm_size = args_info->commsize_arg;
  int NUM_RUNS = 5;
  Goal goal(args_info, comm_size);

  for (int src_rank = 0; src_rank < comm_size; src_rank++) {
    goal.StartRank(src_rank);
    int oldmarker = -1;
    int nops = 0; // running count of colls for collective tag matching
    for (int i = 0; i < NUM_RUNS; i++) {
      goal.Comment("Iallreduce begin");
      goal.SetTag(MAKE_TAG(comm, (collsbase + nops)));
      goal.StartOp();
      create_dissemination_rank(&goal, src_rank, comm_size, 10000);
      std::pair<Goal::locop, Goal::locop> ops = goal.EndOp();
      Goal::locop::iterator it;
      goal.Comment("Iallreduce end");
      nops++;
      int marker = goal.Send(0, 9999999);
      if (oldmarker != -1) {
        for (it = ops.first.begin(); it != ops.first.end(); it++) {
          goal.Requires(it->first, oldmarker);
        }
      }
      for (it = ops.second.begin(); it != ops.second.end(); it++) {
        goal.Requires(marker, it->first);
      }
      oldmarker = marker;
    }
    goal.EndRank();
  }
  goal.Write();
}

namespace {

void recursive_doubling_allreduce(Goal *goal, int rank, int comm_size,
                                  int datasize, const CollectiveContext &ctx) {
  create_allreduce_recdoub_rank(goal, rank, comm_size, datasize);
}

void ring_allreduce(Goal *goal, int rank, int comm_size, int datasize,
                    const CollectiveContext &ctx) {
  create_allreduce_ring_rank(goal, rank, comm_size, datasize);
}

void reduce_scatter_allgather_allreduce(Goal *goal, int rank, int comm_size,
                                        int datasize,
                                        const CollectiveContext &ctx) {
  create_reduce_scatter_allgather_rank(goal, rank, comm_size, datasize,
                                       ctx.replace_comp_time);
}

bool ring_allreduce_validator(int comm_size, int datasize,
                              const CollectiveContext &ctx,
                              std::string *reason) {
  (void)datasize;
  if (ctx.element_count > 0 && ctx.element_count < comm_size) {
    if (reason) {
      *reason = "element count " + std::to_string(ctx.element_count) +
                " smaller than communicator size " +
                std::to_string(comm_size);
    }
    return false;
  }
  return true;
}

bool register_algorithms() {
  register_collective_algorithm(CollectiveKind::Allreduce,
                                "recursive_doubling",
                                recursive_doubling_allreduce);
  register_collective_algorithm(CollectiveKind::Allreduce, "ring",
                                ring_allreduce, ring_allreduce_validator);
  register_collective_algorithm(CollectiveKind::Allreduce,
                                "reduce_scatter_allgather",
                                reduce_scatter_allgather_allreduce);
  return true;
}

const bool registered = register_algorithms();

} // namespace
