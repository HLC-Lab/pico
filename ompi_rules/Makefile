# Compiler and flags
CC = gcc
CFLAGS = -g -Wall

# Colors for output styling
RED := \033[1;31m
YELLOW := \033[1;33m
NC := \033[0m

ifeq ($(OMPI_TEST), 1)
	CFLAGS += -DOMPI_TEST
endif

# Directories
SRC_FILE = change_dynamic_rules.c
BIN_DIR = ../bin

# Output
OUTPUT = $(BIN_DIR)/change_dynamic_rules

# Default target
all: $(OUTPUT)

# Build the executable
$(OUTPUT): $(SRC_FILE)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(CC) $(CFLAGS) $< -o $@

# Clean target
clean:
	@echo -e "$(RED)[CLEAN] Removing ompi_rules binary...$(NC)"
	rm -rf $(OUTPUT)
