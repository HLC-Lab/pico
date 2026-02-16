#include "support_kernel.h"

#define MAX_THERAD 1024
#define GLOBAL_IDX (blockIdx.x * blockDim.x + threadIdx.x)

#define MAKE_KERNEL_OP(type, name, OP)                                                             \
  __global__ void name(type *inbuff, type *outbuff, const type *currentbuff, int comm_size, int groups) \
  {                                                                                                \
    __shared__ type support_buff[MAX_THERAD];                                                      \
    int gidx = GLOBAL_IDX, lidx = threadIdx.x;                                                     \
    int offset = 0;                                                                                \
    if (gidx < comm_size)                                                                               \
    {                                                                                              \
      if (groups > 1)                                                                              \
      {                                                                                            \
        support_buff[lidx] = currentbuff[gidx];                                                    \
        __syncthreads();                                                                           \
        for (int i = 0; i < groups; i++)                                                           \
        {                                                                                          \
          support_buff[lidx] = OP(inbuff[gidx + offset], support_buff[lidx]);                      \
          offset += comm_size;                                                                          \
        }                                                                                          \
        outbuff[gidx] = support_buff[lidx];                                                        \
      }                                                                                            \
      else                                                                                         \
      {                                                                                            \
        outbuff[gidx] = OP(inbuff[gidx], currentbuff[gidx]);                                       \
      }                                                                                            \
    }                                                                                              \
  }

#define MAX_OP(a, b) ((a) > (b) ? (a) : (b))
#define MIN_OP(a, b) ((a) < (b) ? (a) : (b))
#define SUM_OP(a, b) ((a) + (b))
#define MUL_OP(a, b) ((a) * (b))
#define LAND_OP(a, b) ((a) && (b))
#define LOR_OP(a, b) ((a) || (b))
#define LXOR_OP(a, b) ((a) != (b))
#define BAND_OP(a, b) ((a) & (b))
#define BOR_OP(a, b) ((a) | (b))
#define BXOR_OP(a, b) ((a) ^ (b))

#define MAKE_KERNEL_PERM(type, name)                                          \
  __global__ void name(type *inbuff, type *outbuff, int count, int comm_size, int gpu_on_node) \
  {                                                                           \
    int gidx = GLOBAL_IDX;                                                    \
    int node_size = comm_size / gpu_on_node;                                  \
    int elem = gidx % count;                                                  \
    int rank = gidx / count;                                                  \
    int local_rank = rank / node_size;                                        \
    int node_rank = rank % node_size;                                         \
    int dest = (node_rank * gpu_on_node + local_rank) * count + elem;         \
    outbuff[dest] = inbuff[gidx];                                             \
  }

// int8
MAKE_KERNEL_OP(int8_t, sum_int8, SUM_OP)
MAKE_KERNEL_OP(int8_t, prod_int8, MUL_OP)
MAKE_KERNEL_OP(int8_t, max_int8, MAX_OP)
MAKE_KERNEL_OP(int8_t, min_int8, MIN_OP)
MAKE_KERNEL_OP(int8_t, land_int8, LAND_OP)
MAKE_KERNEL_OP(int8_t, band_int8, BAND_OP)
MAKE_KERNEL_OP(int8_t, lor_int8, LOR_OP)
MAKE_KERNEL_OP(int8_t, bor_int8, BOR_OP)
MAKE_KERNEL_OP(int8_t, lxor_int8, LXOR_OP)
MAKE_KERNEL_OP(int8_t, bxor_int8, BXOR_OP)
MAKE_KERNEL_PERM(int8_t, reorder_int8)

// int16
MAKE_KERNEL_OP(int16_t, sum_int16, SUM_OP)
MAKE_KERNEL_OP(int16_t, prod_int16, MUL_OP)
MAKE_KERNEL_OP(int16_t, max_int16, MAX_OP)
MAKE_KERNEL_OP(int16_t, min_int16, MIN_OP)
MAKE_KERNEL_OP(int16_t, land_int16, LAND_OP)
MAKE_KERNEL_OP(int16_t, band_int16, BAND_OP)
MAKE_KERNEL_OP(int16_t, lor_int16, LOR_OP)
MAKE_KERNEL_OP(int16_t, bor_int16, BOR_OP)
MAKE_KERNEL_OP(int16_t, lxor_int16, LXOR_OP)
MAKE_KERNEL_OP(int16_t, bxor_int16, BXOR_OP)
MAKE_KERNEL_PERM(int16_t, reorder_int16)

// int32
MAKE_KERNEL_OP(int32_t, sum_int32, SUM_OP)
MAKE_KERNEL_OP(int32_t, prod_int32, MUL_OP)
MAKE_KERNEL_OP(int32_t, max_int32, MAX_OP)
MAKE_KERNEL_OP(int32_t, min_int32, MIN_OP)
MAKE_KERNEL_OP(int32_t, land_int32, LAND_OP)
MAKE_KERNEL_OP(int32_t, band_int32, BAND_OP)
MAKE_KERNEL_OP(int32_t, lor_int32, LOR_OP)
MAKE_KERNEL_OP(int32_t, bor_int32, BOR_OP)
MAKE_KERNEL_OP(int32_t, lxor_int32, LXOR_OP)
MAKE_KERNEL_OP(int32_t, bxor_int32, BXOR_OP)
MAKE_KERNEL_PERM(int32_t, reorder_int32)

// int64
MAKE_KERNEL_OP(int64_t, sum_int64, SUM_OP)
MAKE_KERNEL_OP(int64_t, prod_int64, MUL_OP)
MAKE_KERNEL_OP(int64_t, max_int64, MAX_OP)
MAKE_KERNEL_OP(int64_t, min_int64, MIN_OP)
MAKE_KERNEL_OP(int64_t, land_int64, LAND_OP)
MAKE_KERNEL_OP(int64_t, band_int64, BAND_OP)
MAKE_KERNEL_OP(int64_t, lor_int64, LOR_OP)
MAKE_KERNEL_OP(int64_t, bor_int64, BOR_OP)
MAKE_KERNEL_OP(int64_t, lxor_int64, LXOR_OP)
MAKE_KERNEL_OP(int64_t, bxor_int64, BXOR_OP)
MAKE_KERNEL_PERM(int64_t, reorder_int64)

// int
MAKE_KERNEL_OP(int, sum_int, SUM_OP)
MAKE_KERNEL_OP(int, prod_int, MUL_OP)
MAKE_KERNEL_OP(int, max_int, MAX_OP)
MAKE_KERNEL_OP(int, min_int, MIN_OP)
MAKE_KERNEL_OP(int, land_int, LAND_OP)
MAKE_KERNEL_OP(int, band_int, BAND_OP)
MAKE_KERNEL_OP(int, lor_int, LOR_OP)
MAKE_KERNEL_OP(int, bor_int, BOR_OP)
MAKE_KERNEL_OP(int, lxor_int, LXOR_OP)
MAKE_KERNEL_OP(int, bxor_int, BXOR_OP)
MAKE_KERNEL_PERM(int, reorder_int)

// float
MAKE_KERNEL_OP(float, sum_float, SUM_OP)
MAKE_KERNEL_OP(float, prod_float, MUL_OP)
MAKE_KERNEL_OP(float, max_float, MAX_OP)
MAKE_KERNEL_OP(float, min_float, MIN_OP)
MAKE_KERNEL_OP(float, land_float, LAND_OP)
MAKE_KERNEL_OP(float, lor_float, LOR_OP)
MAKE_KERNEL_OP(float, lxor_float, LXOR_OP)
MAKE_KERNEL_PERM(float, reorder_float)

// double
MAKE_KERNEL_OP(double, sum_double, SUM_OP)
MAKE_KERNEL_OP(double, prod_double, MUL_OP)
MAKE_KERNEL_OP(double, max_double, MAX_OP)
MAKE_KERNEL_OP(double, min_double, MIN_OP)
MAKE_KERNEL_OP(double, land_double, LAND_OP)
MAKE_KERNEL_OP(double, lor_double, LOR_OP)
MAKE_KERNEL_OP(double, lxor_double, LXOR_OP)
MAKE_KERNEL_PERM(double, reorder_double)

// char
MAKE_KERNEL_OP(char, sum_char, SUM_OP)
MAKE_KERNEL_OP(char, prod_char, MUL_OP)
MAKE_KERNEL_OP(char, max_char, MAX_OP)
MAKE_KERNEL_OP(char, min_char, MIN_OP)
MAKE_KERNEL_OP(char, land_char, LAND_OP)
MAKE_KERNEL_OP(char, band_char, BAND_OP)
MAKE_KERNEL_OP(char, lor_char, LOR_OP)
MAKE_KERNEL_OP(char, bor_char, BOR_OP)
MAKE_KERNEL_OP(char, lxor_char, LXOR_OP)
MAKE_KERNEL_OP(char, bxor_char, BXOR_OP)
MAKE_KERNEL_PERM(char, reorder_char)

typedef void (*kernel_func_op)(void *, void *, const void *, int, int);
typedef void (*kernel_func_reorder)(void *, void *, int, int, int);

static inline enum ReduceOp mpi_to_reduce_op(MPI_Op op)
{
  if (MPI_MAX == op)
    return R_MAX;
  if (MPI_MIN == op)
    return R_MIN;
  if (MPI_SUM == op)
    return R_SUM;
  if (MPI_PROD == op)
    return R_PROD;
  if (MPI_LAND == op)
    return R_LAND;
  if (MPI_BAND == op)
    return R_BAND;
  if (MPI_LOR == op)
    return R_LOR;
  if (MPI_BOR == op)
    return R_BOR;
  if (MPI_LXOR == op)
    return R_LXOR;
  if (MPI_BXOR == op)
    return R_BXOR;
  return R_UNNOWN_OP;
}

static inline enum ReduceType mpi_to_redcue_type(MPI_Datatype dtype)
{
  if (MPI_INT8_T == dtype)
    return R_INT8;
  if (MPI_INT16_T == dtype)
    return R_INT16;
  if (MPI_INT32_T == dtype)
    return R_INT32;
  if (MPI_INT64_T == dtype)
    return R_INT64;
  if (MPI_INT == dtype)
    return R_INT;
  if (MPI_FLOAT == dtype)
    return R_FLOAT;
  if (MPI_DOUBLE == dtype)
    return R_DOUBLE;
  if (MPI_CHAR == dtype)
    return R_CHAR;
  return R_UNNOWN_TYPE;
}

kernel_func_op op_kernels[R_TYPE_NUM][R_OP_NUM] = {
    {(kernel_func_op)sum_int8, (kernel_func_op)prod_int8, (kernel_func_op)max_int8, (kernel_func_op)min_int8, (kernel_func_op)land_int8, (kernel_func_op)band_int8,
     (kernel_func_op)lor_int8, (kernel_func_op)bor_int8, (kernel_func_op)lxor_int8, (kernel_func_op)bxor_int8, NULL}, // int8
    {(kernel_func_op)sum_int16, (kernel_func_op)prod_int16, (kernel_func_op)max_int16, (kernel_func_op)min_int16, (kernel_func_op)land_int16, (kernel_func_op)band_int16,
     (kernel_func_op)lor_int16, (kernel_func_op)bor_int16, (kernel_func_op)lxor_int16, (kernel_func_op)bxor_int16, NULL}, // int16
    {(kernel_func_op)sum_int32, (kernel_func_op)prod_int32, (kernel_func_op)max_int32, (kernel_func_op)min_int32, (kernel_func_op)land_int32, (kernel_func_op)band_int32,
     (kernel_func_op)lor_int32, (kernel_func_op)bor_int32, (kernel_func_op)lxor_int32, (kernel_func_op)bxor_int32, NULL}, // int32
    {(kernel_func_op)sum_int64, (kernel_func_op)prod_int64, (kernel_func_op)max_int64, (kernel_func_op)min_int64, (kernel_func_op)land_int64, (kernel_func_op)band_int64,
     (kernel_func_op)lor_int64, (kernel_func_op)bor_int64, (kernel_func_op)lxor_int64, (kernel_func_op)bxor_int64, NULL}, // int64
    {(kernel_func_op)sum_int, (kernel_func_op)prod_int, (kernel_func_op)max_int, (kernel_func_op)min_int, (kernel_func_op)land_int, (kernel_func_op)band_int, (kernel_func_op)lor_int,
     (kernel_func_op)bor_int, (kernel_func_op)lxor_int, (kernel_func_op)bxor_int, NULL}, // int
    {(kernel_func_op)sum_float, (kernel_func_op)prod_float, (kernel_func_op)max_float, (kernel_func_op)min_float, (kernel_func_op)land_float, (kernel_func_op)NULL,
     (kernel_func_op)lor_float, NULL, (kernel_func_op)lxor_float, NULL, NULL}, // float
    {(kernel_func_op)sum_double, (kernel_func_op)prod_double, (kernel_func_op)max_double, (kernel_func_op)min_double, (kernel_func_op)land_double, (kernel_func_op)NULL,
     (kernel_func_op)lor_double, NULL, (kernel_func_op)lxor_double, NULL, NULL}, // double
    {(kernel_func_op)sum_char, (kernel_func_op)prod_char, (kernel_func_op)max_char, (kernel_func_op)min_char, (kernel_func_op)land_char, (kernel_func_op)band_char,
     (kernel_func_op)lor_char, (kernel_func_op)bor_char, (kernel_func_op)lxor_char, (kernel_func_op)bxor_char, NULL}, // char
    {NULL}                                                                                                            // unnown type
};

kernel_func_reorder reorderute_kernels[R_TYPE_NUM] = {
    (kernel_func_reorder)reorder_int8,
    (kernel_func_reorder)reorder_int16,
    (kernel_func_reorder)reorder_int32,
    (kernel_func_reorder)reorder_int64,
    (kernel_func_reorder)reorder_int,
    (kernel_func_reorder)reorder_float,
    (kernel_func_reorder)reorder_double,
    (kernel_func_reorder)reorder_char,
    NULL
  };

int reduce_wrapper(void *inbuff, void *inoutbuff, int count, MPI_Datatype dtype, MPI_Op op)
{
  // call reduce_wrapper_grops_inoutsplit with sanem value for both outbuff and currentbuff with group comm_size to count and 1 group
  return reduce_wrapper_grops_inoutsplit(inbuff, inoutbuff, inoutbuff, count, 1, dtype, op);
}

int reduce_wrapper_grops(void *inbuff, void *inoutbuff, int group_size, int groups, MPI_Datatype dtype, MPI_Op op)
{
  // call reduce_wrapper_grops_inoutsplit with sanem value for both outbuff and currentbuff
  return reduce_wrapper_grops_inoutsplit(inbuff, inoutbuff, inoutbuff, group_size, groups, dtype, op);
}

int reduce_wrapper_grops_inoutsplit(void *inbuff, void *outbuff, const void *currentbuff, int group_size, int groups, MPI_Datatype dtype, MPI_Op op)
{
  enum ReduceOp r_op = mpi_to_reduce_op(op);
  enum ReduceType r_type = mpi_to_redcue_type(dtype);

  if (r_op == R_UNNOWN_OP || r_type == R_UNNOWN_TYPE || r_op >= R_OP_NUM || r_type >= R_TYPE_NUM)
  {
    return MPI_ERR_UNKNOWN;
  }

  int blockSize = group_size < MAX_THERAD ? group_size : MAX_THERAD;
  int gridSize = (group_size + blockSize - 1) / blockSize;

  kernel_func_op kfunc = op_kernels[r_type][r_op];
  if (kfunc == NULL)
    return MPI_ERR_UNSUPPORTED_OPERATION;

  kfunc<<<gridSize, blockSize>>>(inbuff, outbuff, currentbuff, group_size, groups);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  return MPI_SUCCESS;
}

int reorder_kernel_wrapper(void *inbuff, void *outbuff, int elem, int comm_size, int gpu_on_node, MPI_Datatype dtype)
{
  int total_elem = comm_size * elem;
  enum ReduceType r_type = mpi_to_redcue_type(dtype);
  int blockSize = total_elem < MAX_THERAD ? total_elem : MAX_THERAD;
  int gridSize = (total_elem + blockSize - 1) / blockSize;

  kernel_func_reorder kfunc = reorderute_kernels[r_type];
  if (kfunc == NULL)
    return MPI_ERR_UNSUPPORTED_OPERATION;

  kfunc<<<gridSize, blockSize>>>(inbuff, outbuff, elem, comm_size, gpu_on_node);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  return MPI_SUCCESS;
}