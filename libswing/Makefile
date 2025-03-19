CFLAGS = ${CFLAGS_COMMON}

LIB_DIR = ../lib
LIB_DIR_CUDA = ../lib_cuda
LIB_SRC = $(wildcard *.c)

LIB_OBJ_DIR = ../obj/lib
LIB_OBJS = $(patsubst %.c, $(LIB_OBJ_DIR)/%.o, $(LIB_SRC))
LIB_DEPS = $(patsubst %.c, $(LIB_OBJ_DIR)/%.d, $(LIB_SRC))
LIBSWING = $(LIB_DIR)/libswing.a

LIB_OBJ_DIR_CUDA = ../obj/lib_cuda
LIB_OBJS_CUDA = $(patsubst %.c, $(LIB_OBJ_DIR_CUDA)/%.o, $(LIB_SRC))
LIB_DEPS_CUDA = $(patsubst %.c, $(LIB_OBJ_DIR_CUDA)/%.d, $(LIB_SRC))
LIBSWING_CUDA = $(LIB_DIR_CUDA)/libswing_cuda.a

# Default target
ifeq ($(CUDA_AWARE),1)
all: $(LIBSWING) $(LIBSWING_CUDA)
else
all: $(LIBSWING)
endif

######################################################################
######################STANDARD MAKEFILE TARGETS#######################
######################################################################
$(LIBSWING): $(LIB_OBJS)
	@mkdir -p $(LIB_DIR)
	@echo -e "$(GREEN)[AR] Creating static library: $(LIBSWING)$(NC)"
	ar rcs $@ $^

$(LIB_OBJ_DIR)/%.o: %.c
	@mkdir -p $(LIB_OBJ_DIR)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(CC) $(CFLAGS) -c $< -o $@

-include $(LIB_DEPS)

######################################################################
######################CUDA-AWARE MAKEFILE TARGETS#####################
######################################################################
$(LIBSWING_CUDA): $(LIB_OBJS_CUDA)
	@mkdir -p $(LIB_DIR_CUDA)
	@echo -e "$(GREEN)[AR] Creating static library: $(LIBSWING_CUDA)$(NC)"
	ar rcs $@ $^

$(LIB_OBJ_DIR_CUDA)/%.o: %.c
	@mkdir -p $(LIB_OBJ_DIR_CUDA)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(CC) $(CFLAGS) -DCUDA_AWARE -ldcudart -c $< -o $@

-include $(LIB_DEPS_CUDA)

clean:
	@echo -e "[CLEAN] Removing object files and library..."
	@rm -rf $(LIB_OBJ_DIR) $(LIBSWING) $(LIB_OBJ_DIR_CUDA) $(LIBSWING_CUDA)
