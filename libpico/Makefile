CFLAGS = $(CFLAGS_COMMON)

LIB_DIR        ?= ../lib
LIB_SRC        := $(wildcard *.c)

LIB_OBJ_DIR    ?= ../obj/lib
LIB_OBJS       := $(patsubst %.c, $(LIB_OBJ_DIR)/%.o, $(LIB_SRC))
LIB_DEPS       := $(patsubst %.c, $(LIB_OBJ_DIR)/%.d, $(LIB_SRC))
LIBPICO        := $(LIB_DIR)/libpico.a

LIB_OBJ_DIR_CUDA ?= ../obj/lib_cuda
LIB_OBJS_CUDA    := $(patsubst %.c, $(LIB_OBJ_DIR_CUDA)/%.o, $(LIB_SRC))
LIB_DEPS_CUDA    := $(patsubst %.c, $(LIB_OBJ_DIR_CUDA)/%.d, $(LIB_SRC))
LIBPICO_CUDA     := $(LIB_DIR)/libpico_cuda.a

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
	@echo -e "$(GREEN)[AR] Creating static library: $(LIBPICO)$(NC)"
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

$(LIB_OBJ_DIR_CUDA)/%.o: %.c
	@mkdir -p $(LIB_OBJ_DIR_CUDA)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(PICOCC) $(CFLAGS) -DPICO_MPI_CUDA_AWARE -ldcudart -c $< -o $@

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
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(PICOCC) $(CFLAGS) $(LDLIBS) $(NCCL_FLAG) -DPICO_NCCL -ldcudart -c $< -o $@

-include $(LIB_DEPS_NCCL)


clean:
	@echo -e "[CLEAN] Removing object files and library..."
	@rm -rf $(LIB_OBJ_DIR) $(LIBPICO) $(LIB_OBJ_DIR_CUDA) $(LIBPICO_CUDA) $(LIB_OBJ_DIR_NCCL) $(LIBPICO_NCCL)

