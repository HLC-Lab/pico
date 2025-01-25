# Compiler and flags
MPICC = mpicc
CFLAGS = -O3 -Wall -I../include

ifeq ($(OMPI_TEST), 1)
	CFLAGS += -DOMPI_TEST
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
	@echo -e "\033[1;33m[LINK] Linking $@\033[0m"
	$(MPICC) $(CFLAGS) -o $@ $^ -L../lib -lswing -lm

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	@echo -e "\033[1;33m[MPICC] Compiling $< -> $@\033[0m"
	$(MPICC) $(CFLAGS) -c $< -o $@

# Clean target
clean:
	@echo -e "\033[1;31m[CLEAN] Removing test object files and binary...\033[0m"
	rm -rf $(OBJ_DIR) $(OUTPUT)
