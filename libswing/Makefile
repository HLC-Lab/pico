# Compiler and flags
MPICC = mpicc
CFLAGS = -O3 -Wall -I../include

# Colors for output styling
GREEN := \033[1;32m
YELLOW := \033[1;33m
NC := \033[0m

# Directories
OBJ_DIR = ../obj/lib
LIB_DIR = ../lib

# Sources and objects
LIB_SRC = $(wildcard *.c)
LIB_OBJS = $(patsubst %.c, $(OBJ_DIR)/%.o, $(LIB_SRC))
STATIC_LIB = $(LIB_DIR)/libswing.a

# Default target
all: $(STATIC_LIB)

# Build static library
$(STATIC_LIB): $(LIB_OBJS)
	@mkdir -p $(LIB_DIR)
	@echo -e "$(GREEN)[AR] Creating static library: $(STATIC_LIB)$(NC)"
	@ar rcs $@ $^

# Compile object files
$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(OBJ_DIR)
	@echo -e "$(YELLOW)[MPICC] Compiling $< -> $@...$(NC)"
	@$(MPICC) $(CFLAGS) -c $< -o $@

# Clean target
clean:
	@echo -e "[CLEAN] Removing object files and library..."
	@rm -rf $(OBJ_DIR) $(STATIC_LIB)
