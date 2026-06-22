/*
 * Copyright (c) 2009 The Trustees of Indiana University and Indiana
 *                    University Research and Technology
 *                    Corporation.  All rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *            Timo Schneider <timoschn@cs.indiana.edu>
 *
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "schedgen.hpp"

Goal::Goal(gengetopt_args_info *args_info, int nranks)
    : sends(0), recvs(0), execs(0), ranks(0), reqs(0), curtag(-1) {
  // create filename
  this->filename.clear();
  this->filename.append(std::string(args_info->filename_arg));
  this->myfile.open(this->filename.c_str(),
                    std::fstream::out | std::fstream::trunc);
  std::stringstream tmp;
  tmp << "num_ranks " << nranks << "\n";
  this->AppendString(tmp.str());

  this->cpu = args_info->cpu_arg;
  this->nb = args_info->nb_given;
  if (nb) {
    this->poll_int = args_info->nb_poll_arg;
    this->nbfunc = args_info->nb_arg;
    this->ranks_init.resize(nranks);
    std::fill(this->ranks_init.begin(), this->ranks_init.end(), false);
  }
  this->comm = new Comm();
}

Goal::~Goal() {

  if (myfile.is_open()) {
    myfile.close();
  } else {
    std::cout << "Unable to open file " << this->filename;
    std::cout << " for writing" << std::endl;
  }

  std::cout << sends << " sends, " << recvs << " recvs, " << execs
            << " execs, and " << reqs << " reqs among " << ranks
            << " hosts written" << std::endl;
}

void Goal::StartRank(int rank) {

  this->ranks++;
  this->curtag = 0;

  std::stringstream tmp;
  tmp << "\nrank " << rank << " {\n";
  AppendString(tmp.str());

  // reset label counter
  this->id_counter = 0;

  if (nb) {
    if (!ranks_init[rank]) {
      if (poll_int) { // do we segment?
        int last = 0;
        for (int i = 0; i < nbfunc; i += poll_int) {
          int cur = this->Exec("nbfunc", poll_int, cpu);
          // no last in first round :)
          if (i)
            this->Requires(cur, last);
          last = cur;
        }
      } else { // no segmentation
        this->Exec("nbfunc", nbfunc, cpu);
      }
    }
    ranks_init[rank] = true;
  }
}

Goal::t_id Goal::Send(int bufsize, int dest) {

  this->sends++;

  std::stringstream tmp;

  this->id_counter++;
  tmp << "l" << this->id_counter << ": ";
  tmp << "send ";
  tmp << bufsize << "b ";
  tmp << "to " << dest << " tag " << curtag;
  tmp << std::endl;
  AppendString(tmp.str());

  // append to independent set
  start.insert(id_counter);
  end.insert(id_counter);

  return this->id_counter;
}

void Goal::Comment(std::string comment) {
  std::stringstream tmp;
  tmp << "/* " << comment << " */" << std::endl;
  AppendString(tmp.str());
}

Goal::t_id Goal::Recv(int bufsize, int src) {

  this->recvs++;

  std::stringstream tmp;

  this->id_counter++;
  tmp << "l" << this->id_counter << ": ";
  tmp << "recv ";
  tmp << bufsize << "b ";
  tmp << "from " << src << " tag " << curtag;
  tmp << std::endl;
  AppendString(tmp.str());

  // append to independent set
  start.insert(id_counter);
  end.insert(id_counter);

  return this->id_counter;
}

void Goal::Requires(Goal::t_id tail, Goal::t_id head) {

  this->reqs++;

  std::stringstream tmp;

  tmp << "l" << tail << " requires ";
  tmp << "l" << head << std::endl;

  // erase from independent set
  start.erase(tail);
  end.erase(head);

  AppendString(tmp.str());
}

void Goal::Irequires(int tail, int head) {

  this->reqs++;

  std::stringstream tmp;

  tmp << "l" << tail << " irequires ";
  tmp << "l" << head << std::endl;

  // append to independent set
  start.erase(tail);
  end.erase(head);

  AppendString(tmp.str());
}

void Goal::EndRank() { AppendString("}\n"); }

void Goal::AppendString(std::string str) {
  this->schedule.append(str);

  if (this->schedule.length() > 1024 * 1024 * 16) {
    // write the schedule if it is bigger than 16 MB
    this->Write();
  }
}

void Goal::Write() {

  if (myfile.is_open()) {
    myfile << this->schedule;
    this->schedule.clear();
    myfile.sync();
  } else {
    std::cout << "Unable to open file " << this->filename;
    std::cout << " for writing" << std::endl;
  }
}

// this exec stuff is a big mess and needs to be cleaned up sometime ...
int Goal::Exec(std::string opname, btime_t size, int proc) {

  this->execs++;

  std::stringstream tmp;

  this->id_counter++;
  tmp << "l" << this->id_counter << ": ";
  tmp << "calc " << size;
  if (cpu)
    tmp << " cpu " << proc;
  tmp << std::endl;
  this->schedule.append(tmp.str());

  return this->id_counter;
}

int Goal::Exec(std::string opname, std::vector<buffer_element> buf) {

  this->execs++;

  std::stringstream tmp;

  this->id_counter++;
  tmp << "l" << this->id_counter << ": ";
  tmp << "calc ";
  int size = 0;
  for (unsigned int i = 0; i < buf.size(); i++) {
    size += buf[i].size;
  }
  tmp << size << std::endl;
  this->schedule.append(tmp.str());

  return this->id_counter;
}

int Goal::Exec(std::string opname, btime_t size) {

  std::vector<buffer_element> buf;
  buffer_element elem = buffer_element(1, 1, size);
  buf.push_back(elem);
  return Exec(opname, buf);
}

int main(int argc, char **argv) {

  gengetopt_args_info args_info;

  if (cmdline_parser(argc, argv, &args_info) != 0) {
    fprintf(stderr, "Couldn't parse command line arguments!\n");
    exit(EXIT_FAILURE);
  }

  if (strcmp(args_info.ptrn_arg, "binarytreebcast") == 0) {
    create_binary_tree_bcast(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "binomialtreebcast") == 0) {
    create_binomial_tree_bcast(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "binomialtreereduce") == 0) {
    create_binomial_tree_reduce(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "nwaydissemination") == 0) {
    create_nway_dissemination(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "pipelinedring") == 0) {
    create_pipelined_ring(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "pipelinedringdep") == 0) {
    create_pipelined_ring_dep(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "doublering") == 0) {
    create_double_ring(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "gather") == 0) {
    create_gather(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "scatter") == 0) {
    create_scatter(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "linbarrier") == 0) {
    create_linbarrier(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "dissemination") == 0) {
    create_dissemination(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "random_bisect") == 0) {
    create_random_bisect(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "random_bisect_fd_sym") == 0) {
    create_random_bisect_fd_sym(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "linear_alltoall") == 0) {
    create_linear_alltoall(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "linear_alltoallv") == 0) {
    create_linear_alltoallv(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "allreduce_recdoub") == 0) {
    create_allreduce_recdoub(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "allreduce_ring") == 0) {
    create_allreduce_ring(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "resnet152") == 0) {
    create_resnet152(&args_info);
  }
  if (strcmp(args_info.ptrn_arg, "chained_dissem") == 0) {
    create_chained_dissem(&args_info);
  }

  if (strcmp(args_info.ptrn_arg, "trace") == 0) {
    // see process_trace.cpp
    process_trace(&args_info);
  }

  cmdline_parser_free(&args_info);
  exit(EXIT_SUCCESS);
}
