# Compiler and flags
CFLAGS = $(CFLAGS_COMMON)

# Directories
BIN_DIR = ../bin
BENCH_SRC = $(wildcard *.c)

BENCH_OBJ_DIR = ../obj/bench
BENCH_OBJS = $(patsubst %.c, $(BENCH_OBJ_DIR)/%.o, $(BENCH_SRC))
BENCH_DEPS = $(patsubst %.c, $(BENCH_OBJ_DIR)/%.d, $(BENCH_SRC))
LIB_DIR = ../lib
BENCH = $(BIN_DIR)/bench

BENCH_OBJ_DIR_CUDA = ../obj/bench_cuda
BENCH_OBJS_CUDA = $(patsubst %.c, $(BENCH_OBJ_DIR_CUDA)/%.o, $(BENCH_SRC))
BENCH_DEPS_CUDA = $(patsubst %.c, $(BENCH_OBJ_DIR_CUDA)/%.d, $(BENCH_SRC))
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
$(BENCH): $(BENCH_OBJS)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking bench: $< -> $@$(NC)"
	$(CC) $(CFLAGS) -o $@ $^ -L$(LIB_DIR) -lswing -lm

# Compile source files
$(BENCH_OBJ_DIR)/%.o: %.c
	@mkdir -p $(BENCH_OBJ_DIR)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(CC) $(CFLAGS) -c $< -o $@

-include $(BENCH_DEPS)

$(BENCH_CUDA): $(BENCH_OBJS_CUDA)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking bench_cuda: $< -> $@$(NC)"
	$(CC) $(CFLAGS) -o $@ $^ -L$(LIB_DIR_CUDA) -lswing_cuda -lm


$(BENCH_OBJS_DIR_CUDA)/%.o: %.c
	@mkdir -p $(BENCH_OBJ_DIR_CUDA)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(CC) $(CFLAGS) -DCUDA_AWARE -lcudart -c $< -o $@

-include $(BENCH_DEPS_CUDA)


clean:
	@echo -e "[CLEAN] Removing bench object files and binary..."
	@rm -rf $(BENCH_OBJ_DIR) $(BENCH) $(BENCH_OBJ_DIR_CUDA) $(BENCH_CUDA)

