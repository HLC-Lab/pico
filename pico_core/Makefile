# Compiler and flags
CFLAGS = $(CFLAGS_COMMON)

# Directories
BIN_DIR = ../bin
PICO_CORE_SRC = $(wildcard *.c)

PICO_CORE_OBJ_DIR = ../obj/pico_core
PICO_CORE_OBJS = $(patsubst %.c, $(PICO_CORE_OBJ_DIR)/%.o, $(PICO_CORE_SRC))
PICO_CORE_DEPS = $(patsubst %.c, $(PICO_CORE_OBJ_DIR)/%.d, $(PICO_CORE_SRC))
LIB_DIR = ../lib
PICO_CORE = $(BIN_DIR)/pico_core

PICO_CORE_OBJ_DIR_CUDA = ../obj/pico_core_cuda
PICO_CORE_OBJS_CUDA = $(patsubst %.c, $(PICO_CORE_OBJ_DIR_CUDA)/%.o, $(PICO_CORE_SRC))
PICO_CORE_DEPS_CUDA = $(patsubst %.c, $(PICO_CORE_OBJ_DIR_CUDA)/%.d, $(PICO_CORE_SRC))
PICO_CORE_CUDA = $(BIN_DIR)/pico_core_cuda

ifeq ($(DEBUG),1)
	CFLAGS := $(filter-out -O%,$(CFLAGS))
endif

ifeq ($(CUDA_AWARE),1)
all: $(PICO_CORE_CUDA) $(PICO_CORE)
else
all: $(PICO_CORE)
endif

# Build the executable
$(PICO_CORE): $(PICO_CORE_OBJS)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking pico_core: $< -> $@$(NC)"
	$(BINECC) $(CFLAGS) -o $@ $^ -L$(LIB_DIR) -lbine -lm

# Compile source files
$(PICO_CORE_OBJ_DIR)/%.o: %.c
	@mkdir -p $(PICO_CORE_OBJ_DIR)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(BINECC) $(CFLAGS) -c $< -o $@

-include $(PICO_CORE_DEPS)

$(PICO_CORE_CUDA): $(PICO_CORE_OBJS_CUDA)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking pico_core_cuda: $< -> $@$(NC)"
	$(BINECC) $(CFLAGS) -o $@ $^ -L$(LIB_DIR) -lbine_cuda -lcudart -lm

$(PICO_CORE_OBJ_DIR_CUDA)/%.o: %.c
	@mkdir -p $(PICO_CORE_OBJ_DIR_CUDA)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(BINECC) $(CFLAGS) -DCUDA_AWARE -lcudart -c $< -o $@

-include $(PICO_CORE_DEPS_CUDA)


clean:
	@echo -e "[CLEAN] Removing pico_core object files and binary..."
	@rm -rf $(PICO_CORE_OBJ_DIR) $(PICO_CORE) $(PICO_CORE_OBJ_DIR_CUDA) $(PICO_CORE_CUDA)

