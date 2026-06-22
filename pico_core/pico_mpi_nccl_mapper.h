/*
 * Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
 * Licensed under the MIT License
 */
#ifndef PICO_MPI_NCCL_MAPPER_H
#define PICO_MPI_NCCL_MAPPER_H

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <mpi.h>

#if defined PICO_NCCL || defined PICO_MPI_CUDA_AWARE
#include <cuda_runtime.h>
#endif
#ifdef PICO_NCCL
#include <nccl.h>
#endif



/* ---------------- Selector & helpers ---------------- */
#ifndef PICO_NCCL
#  define PICO_PICK(NCCL_EXPR, MPI_EXPR) MPI_EXPR
#else
#  define PICO_PICK(NCCL_EXPR, MPI_EXPR) NCCL_EXPR
#endif

#if defined PICO_NCCL || defined PICO_MPI_CUDA_AWARE
#define PICO_CORE_CUDA_CHECK(cmd, err) do {             \
  err = cmd;                                            \
  if( err != cudaSuccess ) {                            \
    fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n",  \
        __FILE__, __LINE__, cudaGetErrorString(err));   \
    return -1;                                          \
  }                                                     \
} while(0)
#endif

#ifdef PICO_NCCL
#define PICO_NCCL_CHECK(cmd, err) do {                      \
  err = (cmd);                                              \
  if (err != ncclSuccess) {                                 \
    fprintf(stderr, "NCCL error %s:%d: %s\n",               \
            __FILE__, __LINE__, ncclGetErrorString(err));   \
    return -1;                                              \
  }                                                         \
} while (0)

#endif

#ifdef PICO_NCCL
//-----------------------------------------------------------------------------------------------
//                                INITIALIZE and FINALIZE NCCL
//-----------------------------------------------------------------------------------------------

/**
* @brief Initializes NCCL communicator and CUDA stream for the given MPI communicator.
*
* @param comm MPI communicator.
* @param rank Rank of the calling process in the communicator.
* @param comm_sz Size of the communicator (number of processes).
* @param[out] out_comm Pointer to the NCCL communicator to be initialized.
* @param[out] out_stream Pointer to the CUDA stream to be created.
* @return 0 on success, -1 on failure.
* @note Assumes one visible GPU per MPI rank. Returns an error if not.
*/
static inline int pico_nccl_init(MPI_Comm comm, int rank, int comm_sz,
                                 ncclComm_t* out_comm, cudaStream_t* out_stream)
{
  cudaError_t cuda_err;
  ncclResult_t nccl_err;

  int visible = 0;
  PICO_CORE_CUDA_CHECK(cudaGetDeviceCount(&visible), cuda_err);
  if (visible != 1 && rank == 0) {
    fprintf(stderr,
      "[ERROR] Expected 1 visible GPU per task, but saw %d. "
      "Check your srun flags.\n", visible);
    return -1;
  }

  ncclUniqueId id;
  if (rank == 0) PICO_NCCL_CHECK(ncclGetUniqueId(&id), nccl_err);
  PMPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, comm);

  PICO_NCCL_CHECK(ncclCommInitRank(out_comm, comm_sz, id, rank), nccl_err);
  PICO_CORE_CUDA_CHECK(cudaStreamCreateWithFlags(out_stream, cudaStreamNonBlocking), cuda_err);

  return 0;
}


/**
 * @brief Finalizes NCCL communicator and destroys the CUDA stream.
 *
 * @param comm NCCL communicator to be destroyed.
 * @param stream CUDA stream to be destroyed.
 * @return 0 on success, -1 on failure.
 * @note Assumes that the communicator and stream were previously initialized.
 */
static inline int pico_nccl_finalize(ncclComm_t comm, cudaStream_t stream)
{
  cudaError_t  cuda_err;
  ncclResult_t nccl_err;
  PICO_CORE_CUDA_CHECK(cudaStreamDestroy(stream), cuda_err);
  PICO_NCCL_CHECK(ncclCommDestroy(comm), nccl_err);
  return 0;
}

#endif

/* ---------------- Types ----------------------------------- */
#define PICO_COMM_T         PICO_PICK(ncclComm_t,      MPI_Comm)
#define PICO_RESULT_T       PICO_PICK(ncclResult_t,    int)
#define PICO_DTYPE_T        PICO_PICK(ncclDataType_t, MPI_Datatype)

/* ---------------- Datatypes ----------------------------------- */
#define PICO_DTYPE_INVALID  PICO_PICK((ncclDataType_t)(-1), MPI_DATATYPE_NULL)
#define PICO_DTYPE_INT8     PICO_PICK(ncclInt8,             MPI_INT8_T)
#define PICO_DTYPE_INT16    PICO_PICK(PICO_DTYPE_INVALID,   MPI_INT16_T)
#define PICO_DTYPE_INT32    PICO_PICK(ncclInt32,            MPI_INT32_T)
#define PICO_DTYPE_INT64    PICO_PICK(ncclInt64,            MPI_INT64_T)
#define PICO_DTYPE_INT      PICO_PICK(ncclInt32,            MPI_INT) // for compatibility with old NCCL versions
#define PICO_DTYPE_FLOAT    PICO_PICK(ncclFloat32,          MPI_FLOAT)
#define PICO_DTYPE_DOUBLE   PICO_PICK(ncclFloat64,          MPI_DOUBLE)
#define PICO_DTYPE_CHAR     PICO_PICK(ncclInt8,             MPI_CHAR) // for compatibility with old NCCL versions



/**
 * @struct LayerTypeMap
 * @brief Maps string names to backend datatype tokens and element sizes.
 * @note  'supported' is 0 when this name isn’t available on the chosen backend.
 */
typedef struct {
  const char*     t_string;   /* Type name as a string */
  PICO_DTYPE_T    pico_type; /* Backend datatype token (NCCL or MPI) */
  MPI_Datatype    mpi_type;  /* MPI datatype token (for reference) */
  size_t          t_size;     /* Size of the element in bytes */
} PicoTypeMap;


/**
 * @brief Static array mapping string representations to MPI datatypes. Will be
 *        used to map command-line input argument to datatype and its size.
 */
const static PicoTypeMap type_map[] = {
  {"int8",    PICO_DTYPE_INT8,    MPI_INT8_T,   sizeof(int8_t)},
  {"int16",   PICO_DTYPE_INT16,   MPI_INT16_T,  sizeof(int16_t)},
  {"int32",   PICO_DTYPE_INT32,   MPI_INT32_T,  sizeof(int32_t)},
  {"int64",   PICO_DTYPE_INT64,   MPI_INT64_T,  sizeof(int64_t)},
  {"int",     PICO_DTYPE_INT,     MPI_INT,      sizeof(int)},
  {"float",   PICO_DTYPE_FLOAT,   MPI_FLOAT,    sizeof(float)},
  {"double",  PICO_DTYPE_DOUBLE,  MPI_DOUBLE,   sizeof(double)},
  {"char",    PICO_DTYPE_CHAR,    MPI_CHAR,     sizeof(char)}
};


/**
 * @brief Retrieves the MPI/NCCL datatype and size based on a string identifier utilizing `type_map`.
 *
 * @param type_string String representation of the data type.
 * @param[out] dtype MPI or NCCL datatype corresponding to the string.
 * @param[out] type_size Size of the datatype in bytes.
 * @return 0 on success, -1 if the data type is invalid.
 */
static inline int get_data_type(const char *type_string, PICO_DTYPE_T *dtype, MPI_Datatype *mpi_dtype, size_t *type_size) {
  if (!type_string || !dtype || !type_size) {
    fprintf(stderr, "Error: invalid arguments to get_data_type. Aborting...\n");
    return -1;
  }

  int num_types = sizeof(type_map) / sizeof(type_map[0]);

  for(int i = 0; i < num_types; i++) {
    if(strcmp(type_string, type_map[i].t_string) == 0) {
      if (type_map[i].pico_type == PICO_DTYPE_INVALID) {
        fprintf(stderr, "Error: datatype %s not supported by \
                  the selected backend. Aborting...", type_string);
        return -1;
      }
      *dtype = type_map[i].pico_type;
      *mpi_dtype = type_map[i].mpi_type;
      *type_size = type_map[i].t_size;
      return 0;
    }
  }

  fprintf(stderr, "Error: datatype %s not in `type_map`. Aborting...", type_string);
  return -1;
}


#endif
