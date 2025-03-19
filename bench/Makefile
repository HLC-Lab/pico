# Compiler and flags
CFLAGS = $(CFLAGS_COMMON)

# Directories
SRC_DIR = .
BIN_DIR = ../bin
OBJ_DIR = ../obj/bench
OBJ_DIR_CUDA = ../obj/bench_cuda
LIB_DIR = ../lib

SRC = $(wildcard *.c)
OBJS = $(patsubst %.c, $(OBJ_DIR)/%.o, $(SRC))
DEPS = $(patsubst %.c, $(OBJ_DIR)/%.d, $(SRC))
BENCH = $(BIN_DIR)/bench
OBJS_CUDA = $(patsubst %.c, $(OBJ_DIR_CUDA)/%.o, $(SRC))
DEPS_CUDA = $(patsubst %.c, $(OBJ_DIR_CUDA)/%.d, $(SRC))
BENCH_CUDA = $(BIN_DIR)/bench_cuda

ifeq ($(DEBUG),1)
	CFLAGS := $(filter-out -O%,$(CFLAGS))
endif

all: $(BENCH)
ifeq ($(CUDA_WARE),1)
all: $(BENCH_CUDA)
endif

# Build the executable
$(BENCH): $(OBJS)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking bench: $< -> $@$(NC)"
	$(CC) $(CFLAGS) -o $@ $^ -L$(LIB_DIR) -lswing -lm

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(CC) $(CFLAGS) -c $< -o $@

-include $(DEPS)

$(BENCH_CUDA): $(OBJS_CUDA)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking bench_cuda: $< -> $@$(NC)"
	$(CC) $(CFLAGS) -o $@ $^ -L$(LIB_DIR) -lswing_cuda -lm


$(OBJS_CUDA): $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR_CUDA)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(CC) $(CFLAGS) -DCUDA_AWARE -lcudart -c $< -o $@

-include $(DEPS_CUDA)


clean:
	@echo -e "[CLEAN] Removing bench object files and binary..."
	@rm -rf $(OBJ_DIR) $(BENCH) $(OBJ_DIR_CUDA) $(BENCH_CUDA)

