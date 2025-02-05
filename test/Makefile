# Compiler and flags
MPICC = mpicc
CFLAGS = -O3 -Wall -I../include

# Colors for output styling
GREEN := \033[1;32m
YELLOW := \033[1;33m
RED := \033[1;31m
NC := \033[0m

ifeq ($(DEBUG), 1)
	CFLAGS += -DDEBUG
endif

# Directories
SRC_DIR = .
OBJ_DIR = ../obj/test
BIN_DIR = ../bin

# Sources and output
SRC = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC))
OUTPUT = $(BIN_DIR)/test

# Default target
all: $(OUTPUT)

# Build the executable
$(OUTPUT): $(OBJS)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking $@...$(NC)"
	@$(MPICC) $(CFLAGS) -o $@ $^ -L../lib -lswing -lm

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	@echo -e "$(YELLOW)[MPICC] Compiling $< -> $@...$(NC)"
	@$(MPICC) $(CFLAGS) -c $< -o $@

# Clean target
clean:
	@echo -e "[CLEAN] Removing test object files and binary..."
	@rm -rf $(OBJ_DIR) $(OUTPUT)
