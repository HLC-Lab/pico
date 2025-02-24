# Compiler and flags
CFLAGS = $(CFLAGS_COMMON)

# Directories
SRC_DIR = .
OBJ_DIR = ../obj/test
BIN_DIR = ../bin

# Sources and output
SRC = $(wildcard *.c)
OBJS = $(patsubst %.c, $(OBJ_DIR)/%.o, $(SRC))
DEPS = $(patsubst %.c, $(OBJ_DIR)/%.d, $(SRC))
OUTPUT = $(BIN_DIR)/test

ifeq ($(DEBUG),1)
	CFLAGS := $(filter-out -O%,$(CFLAGS))
	CFLAGS := $(filter-out -ffp=3,$(CFLAGS))
endif

# Default target
all: $(OUTPUT)

# Build the executable
$(OUTPUT): $(OBJS)
	@mkdir -p $(BIN_DIR)
	@echo -e "$(GREEN)[LINK] Linking $@...$(NC)"
	$(CC) $(CFLAGS) -o $@ $^ -L../lib -lswing -lm

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(CC) $(CFLAGS) -c $< -o $@

-include $(DEPS)

# Clean target
clean:
	@echo -e "[CLEAN] Removing test object files and binary..."
	@rm -rf $(OBJ_DIR) $(OUTPUT)
