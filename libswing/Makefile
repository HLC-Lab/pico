CFLAGS = ${CFLAGS_COMMON}

LIB_DIR = ../lib
LIB_SRC = $(wildcard *.c)

OBJ_DIR = ../obj/lib
LIB_OBJS = $(patsubst %.c, $(OBJ_DIR)/%.o, $(LIB_SRC))
LIB_DEPS = $(patsubst %.c, $(OBJ_DIR)/%.d, $(LIB_SRC))
LIBSWING = $(LIB_DIR)/libswing.a

OBJ_DIR_CUDA = ../obj/lib_cuda
LIB_OBJS_CUDA = $(patsubst %.c, $(OBJ_DIR_CUDA)/%.o, $(LIB_SRC))
LIB_DEPS_CUDA = $(patsubst %.c, $(OBJ_DIR_CUDA)/%.d, $(LIB_SRC))
LIBSWING_CUDA = $(LIB_DIR)/libswing_cuda.a

# Default target
all: $(LIBSWING)
ifeq ($(CUDA_WARE),1)
all: $(LIBSWING_CUDA)
endif

######################################################################
######################STANDARD MAKEFILE TARGETS#######################
######################################################################
$(LIBSWING): $(LIB_OBJS)
	@mkdir -p $(LIB_DIR)
	@echo -e "$(GREEN)[AR] Creating static library: $(LIBSWING)$(NC)"
	ar rcs $@ $^

$(OBJ_DIR)/%.o: %.c
	@mkdir -p $(OBJ_DIR)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(CC) $(CFLAGS) -c $< -o $@

-include $(LIB_DEPS)

######################################################################
######################CUDA-AWARE MAKEFILE TARGETS#####################
######################################################################
$(LIBSWING_CUDA): $(LIB_OBJS_CUDA)
	@mkdir -p $(LIB_DIR)
	@echo -e "$(GREEN)[AR] Creating static library: $(LIBSWING_CUDA)$(NC)"
	ar rcs $@ $^

$(OBJ_DIR_CUDA)/%.o: %.c
	@mkdir -p $(OBJ_DIR_CUDA)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(CC) $(CFLAGS) -DCUDA_AWARE -ldcudart -c $< -o $@

-include $(LIB_DEPS_CUDA)

clean:
	@echo -e "[CLEAN] Removing object files and library..."
	@rm -rf $(OBJ_DIR) $(LIBSWING) $(OBJ_DIR_CUDA) $(LIBSWING_CUDA)
