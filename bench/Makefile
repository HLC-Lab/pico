# Compiler and flags
CFLAGS = $(CFLAGS_COMMON)

# Directories
BIN_DIR = ../bin
BENCH = $(BIN_DIR)/bench
BENCH_SRC = $(wildcard *.c)

OBJ_DIR = ../obj/bench
OBJS = $(patsubst %.c, $(OBJ_DIR)/%.o, $(BENCH_SRC))
DEPS = $(patsubst %.c, $(OBJ_DIR)/%.d, $(BENCH_SRC))
LIB_DIR = ../lib

OBJ_DIR_CUDA = ../obj/bench_cuda
OBJS_CUDA = $(patsubst %.c, $(OBJ_DIR_CUDA)/%.o, $(BENCH_SRC))
DEPS_CUDA = $(patsubst %.c, $(OBJ_DIR_CUDA)/%.d, $(BENCH_SRC))
LIB_DIR_CUDA = ../lib_cuda
BENCH_CUDA = $(BIN_DIR)/bench_cuda

ifeq ($(DEBUG),1)
	CFLAGS := $(filter-out -O%,$(CFLAGS))
endif

ifeq ($(CUDA_AWARE),1)
all: $(BENCH_CUDA) $(BENCH)
else
all: $(BENCH)
endif

# Build the executable
$(BENCH): $(OBJS)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking bench: $< -> $@$(NC)"
	$(CC) $(CFLAGS) -o $@ $^ -L$(LIB_DIR) -lswing -lm

# Compile source files
$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(OBJ_DIR)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(CC) $(CFLAGS) -c $< -o $@

-include $(DEPS)

$(BENCH_CUDA): $(OBJS_CUDA)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking bench_cuda: $< -> $@$(NC)"
	$(CC) $(CFLAGS) -o $@ $^ -L$(LIB_DIR_CUDA) -lswing_cuda -lm


$(OBJS_CUDA)/%.o: %.c
	@mkdir -p $(OBJ_DIR_CUDA)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(CC) $(CFLAGS) -DCUDA_AWARE -lcudart -c $< -o $@

-include $(DEPS_CUDA)


clean:
	@echo -e "[CLEAN] Removing bench object files and binary..."
	@rm -rf $(OBJ_DIR) $(BENCH) $(OBJ_DIR_CUDA) $(BENCH_CUDA)

