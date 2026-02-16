CFLAGS = $(CFLAGS_COMMON)
NVCC ?= nvcc

NVCCFLAGS := -O3 -Xcompiler "-fPIC" -I$(PICO_DIR)/include -I.

NVCCFLAGS += -Xcompiler "-MMD -MP"

# Aggiunge i flag condizionali (DEBUG, INSTRUMENT) a NVCC
ifeq ($(DEBUG),1)
    NVCCFLAGS += -DDEBUG -g
endif
ifeq ($(PICO_INSTRUMENT),1)
    NVCCFLAGS += -DPICO_INSTRUMENT
endif

CFLAGS_CUDA =
ifeq ($(GPU_NATIV_SUPPORT),1)
    CFLAGS_CUDA += -DGPU_NATIV_SUPPORT
    NVCCFLAGS += -DGPU_NATIV_SUPPORT
endif
ifeq ($(PICO_MPI_CUDA_AWARE),1)
    CFLAGS_CUDA += -DPICO_MPI_CUDA_AWARE
    NVCCFLAGS += -DPICO_MPI_CUDA_AWARE
endif


LIB_DIR ?= ../lib

LIB_SRC := $(wildcard *.c)
LIB_OBJ_DIR ?= ../obj/lib
LIB_OBJS := $(patsubst %.c,$(LIB_OBJ_DIR)/%.o,$(LIB_SRC))
LIB_DEPS := $(patsubst %.o,%.d,$(LIB_OBJS))
LIBPICO := $(LIB_DIR)/libpico.a

PICO_CORE_OBJ_DIR_CUDA ?= ../obj/lib_cuda
LIBPICO_CUDA := $(LIB_DIR)/libpico_cuda.a

LIB_SRC_C_FILES_CUDA := $(wildcard *.c)
LIB_SRC_CU_FILES_CUDA := $(wildcard *.cu)

LIB_OBJS_C_CUDA := $(patsubst %.c,$(PICO_CORE_OBJ_DIR_CUDA)/%.o,$(LIB_SRC_C_FILES_CUDA))
LIB_DEPS_C_CUDA := $(patsubst %.o,%.d,$(LIB_OBJS_C_CUDA))

LIB_OBJS_CU_CUDA := $(patsubst %.cu,$(PICO_CORE_OBJ_DIR_CUDA)/%.o,$(LIB_SRC_CU_FILES_CUDA))
LIB_DEPS_CU_CUDA := $(patsubst %.o,%.d,$(LIB_OBJS_CU_CUDA))

LIB_OBJS_CUDA := $(LIB_OBJS_C_CUDA) $(LIB_OBJS_CU_CUDA)
LIB_DEPS_CUDA := $(LIB_DEPS_C_CUDA) $(LIB_DEPS_CU_CUDA)


LIB_OBJ_DIR_NCCL ?= ../obj/lib_nccl
LIB_OBJS_NCCL    := $(patsubst %.c, $(LIB_OBJ_DIR_NCCL)/%.o, $(LIB_SRC))
LIB_DEPS_NCCL    := $(patsubst %.c, $(LIB_OBJ_DIR_NCCL)/%.d, $(LIB_SRC))
LIBPICO_NCCL     := $(LIB_DIR)/libpico_nccl.a


ifeq ($(PICO_MPI_CUDA_AWARE),1)
all: $(LIBPICO) $(LIBPICO_CUDA)
else
all: $(LIBPICO)
endif

ifeq ($(PICO_NCCL),1)
all: $(LIBPICO_NCCL)
endif

######################################################################
######################STANDARD MAKEFILE TARGETS#######################
######################################################################
$(LIBPICO): $(LIB_OBJS)
	@mkdir -p $(LIB_DIR)
	@echo "[AR] Creating standard library $@"
	ar rcs $@ $^

$(LIB_OBJ_DIR)/%.o: %.c
	@mkdir -p $(LIB_OBJ_DIR)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(PICOCC) $(CFLAGS) -c $< -o $@

-include $(LIB_DEPS)

######################################################################
######################CUDA-AWARE MAKEFILE TARGETS#####################
######################################################################
$(LIBPICO_CUDA): $(LIB_OBJS_CUDA)
	@mkdir -p $(LIB_DIR)
	@echo -e "$(GREEN)[AR] Creating static library: $(LIBPICO_CUDA)$(NC)"
	ar rcs $@ $^

# Regola per file .c (compilati con PICOCC)
$(PICO_CORE_OBJ_DIR_CUDA)/%.o: %.c
	@mkdir -p $(PICO_CORE_OBJ_DIR_CUDA)
	@echo -e "$(YELLOW)[CC] Compiling CUDA $< -> $@...$(NC)"
	$(PICOCC) $(CFLAGS) $(CFLAGS_CUDA) -c $< -o $@

# Regola per file .cu (compilati con NVCC)
$(PICO_CORE_OBJ_DIR_CUDA)/%.o: %.cu
	@mkdir -p $(PICO_CORE_OBJ_DIR_CUDA)
	@echo -e "$(YELLOW)[CC] Compiling CUDA $< -> $@...$(NC)"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

-include $(LIB_DEPS_CUDA)

######################################################################
######################NCCL-AWARE MAKEFILE TARGETS#####################
######################################################################
$(LIBPICO_NCCL): $(LIB_OBJS_NCCL)
	@mkdir -p $(LIB_DIR)
	@echo -e "$(GREEN)[AR] Creating static library: $(LIBPICO_NCCL)$(NC)"
	ar rcs $@ $^

$(LIB_OBJ_DIR_NCCL)/%.o: %.c
	@mkdir -p $(LIB_OBJ_DIR_NCCL)
	@echo -e "$(YELLOW)[CC] Compiling NCCL $< -> $@...$(NC)"
	$(PICOCC) $(CFLAGS) $(LDLIBS) $(NCCL_FLAG) -DPICO_NCCL -ldcudart -c $< -o $@

-include $(LIB_DEPS_NCCL)


clean:
	@echo -e "[CLEAN] Removing object files and library..."
	@rm -rf $(LIB_OBJ_DIR) $(PICO_CORE_OBJ_DIR_CUDA) $(LIBPICO) $(LIBPICO_CUDA) $(LIB_OBJ_DIR_NCCL) $(LIBPICO_NCCL)