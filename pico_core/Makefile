# Compiler and flags
CFLAGS = $(CFLAGS_COMMON)

BIN_DIR            ?= ../bin
PICO_CORE_SRC      := $(wildcard *.c)

PICO_CORE_OBJ_DIR       ?= ../obj/pico_core
PICO_CORE_OBJS          := $(patsubst %.c, $(PICO_CORE_OBJ_DIR)/%.o, $(PICO_CORE_SRC))
PICO_CORE_DEPS          := $(patsubst %.c, $(PICO_CORE_OBJ_DIR)/%.d, $(PICO_CORE_SRC))
LIB_DIR                 ?= ../lib
PICO_CORE               := $(BIN_DIR)/pico_core

PICO_CORE_OBJ_DIR_CUDA  ?= ../obj/pico_core_cuda
PICO_CORE_OBJS_CUDA     := $(patsubst %.c, $(PICO_CORE_OBJ_DIR_CUDA)/%.o, $(PICO_CORE_SRC))
PICO_CORE_DEPS_CUDA     := $(patsubst %.c, $(PICO_CORE_OBJ_DIR_CUDA)/%.d, $(PICO_CORE_SRC))
PICO_CORE_CUDA          := $(BIN_DIR)/pico_core_cuda
ifeq ($(GPU_NATIV_SUPPORT),1)
CFLAGS_CUDA += -DGPU_NATIV_SUPPORT
endif

PICO_CORE_OBJ_DIR_NCCL  ?= ../obj/pico_core_nccl
PICO_CORE_OBJS_NCCL     := $(patsubst %.c, $(PICO_CORE_OBJ_DIR_NCCL)/%.o, $(PICO_CORE_SRC))
PICO_CORE_DEPS_NCCL     := $(patsubst %.c, $(PICO_CORE_OBJ_DIR_NCCL)/%.d, $(PICO_CORE_SRC))
PICO_CORE_NCCL          := $(BIN_DIR)/pico_core_nccl


ifeq ($(DEBUG),1)
	CFLAGS := $(filter-out -O%,$(CFLAGS))
endif

ifeq ($(PICO_MPI_CUDA_AWARE),1)
all: $(PICO_CORE_CUDA) $(PICO_CORE)
else
all: $(PICO_CORE)
endif

ifeq ($(PICO_NCCL),1)
all: $(PICO_CORE_NCCL)
endif

# Build the executable
$(PICO_CORE): $(PICO_CORE_OBJS)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking pico_core: $< -> $@$(NC)"
	$(PICOCC) $(CFLAGS) -o $@ $^ -L$(LIB_DIR) -lpico -lm

# Compile source files
$(PICO_CORE_OBJ_DIR)/%.o: %.c
	@mkdir -p $(PICO_CORE_OBJ_DIR)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(PICOCC) $(CFLAGS) -c $< -o $@

-include $(PICO_CORE_DEPS)

$(PICO_CORE_CUDA): $(PICO_CORE_OBJS_CUDA)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking pico_core_cuda: $< -> $@$(NC)"
	$(PICOCC) $(CFLAGS) -o $@ $^ -L$(LIB_DIR) -lpico_cuda -lcudart -lstdc++ -lm

$(PICO_CORE_OBJ_DIR_CUDA)/%.o: %.c
	@mkdir -p $(PICO_CORE_OBJ_DIR_CUDA)
	@echo -e "$(YELLOW)[CC] Compiling CUDA $< -> $@...$(NC)"
	$(PICOCC) $(CFLAGS) $(CFLAGS_CUDA) -DPICO_MPI_CUDA_AWARE -lpico_cuda -lcudart -c $< -o $@

-include $(PICO_CORE_DEPS_CUDA)

$(PICO_CORE_NCCL): $(PICO_CORE_OBJS_NCCL)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking pico_core_cuda: $< -> $@$(NC)"
	$(PICOCC) $(CFLAGS) $(LDLIBS) $(NCCL_FLAG)  -o $@ $^ -L$(LIB_DIR) -lpico_nccl -lcudart -lm

$(PICO_CORE_OBJ_DIR_NCCL)/%.o: %.c
	@mkdir -p $(PICO_CORE_OBJ_DIR_NCCL)
	@echo -e "$(YELLOW)[CC] Compiling NCCL $< -> $@...$(NC)"
	$(PICOCC) $(CFLAGS) $(LDLIBS) $(NCCL_FLAG) -DPICO_NCCL -lcudart -c $< -o $@

-include $(PICO_CORE_DEPS_CUDA)

clean:
	@echo -e "[CLEAN] Removing pico_core object files and binary..."
	@rm -rf $(PICO_CORE_OBJ_DIR) $(PICO_CORE) $(PICO_CORE_OBJ_DIR_CUDA) $(PICO_CORE_CUDA) $(PICO_CORE_OBJ_DIR_NCCL) $(PICO_CORE_NCCL)