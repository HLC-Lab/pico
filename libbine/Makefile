CFLAGS = $(CFLAGS_COMMON)

LIB_DIR = ../lib
LIB_SRC = $(wildcard *.c)

LIB_OBJ_DIR = ../obj/lib
LIB_OBJS = $(patsubst %.c, $(LIB_OBJ_DIR)/%.o, $(LIB_SRC))
LIB_DEPS = $(patsubst %.c, $(LIB_OBJ_DIR)/%.d, $(LIB_SRC))
LIBBINE = $(LIB_DIR)/libbine.a

LIB_OBJ_DIR_CUDA = ../obj/lib_cuda
LIB_OBJS_CUDA = $(patsubst %.c, $(LIB_OBJ_DIR_CUDA)/%.o, $(LIB_SRC))
LIB_DEPS_CUDA = $(patsubst %.c, $(LIB_OBJ_DIR_CUDA)/%.d, $(LIB_SRC))
LIBBINE_CUDA = $(LIB_DIR)/libbine_cuda.a

# Default target
ifeq ($(CUDA_AWARE),1)
all: $(LIBBINE) $(LIBBINE_CUDA)
else
all: $(LIBBINE)
endif

######################################################################
######################STANDARD MAKEFILE TARGETS#######################
######################################################################
$(LIBBINE): $(LIB_OBJS)
	@mkdir -p $(LIB_DIR)
	@echo -e "$(GREEN)[AR] Creating static library: $(LIBBINE)$(NC)"
	ar rcs $@ $^

$(LIB_OBJ_DIR)/%.o: %.c
	@mkdir -p $(LIB_OBJ_DIR)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(BINECC) $(CFLAGS) -c $< -o $@

-include $(LIB_DEPS)

######################################################################
######################CUDA-AWARE MAKEFILE TARGETS#####################
######################################################################
$(LIBBINE_CUDA): $(LIB_OBJS_CUDA)
	@mkdir -p $(LIB_DIR)
	@echo -e "$(GREEN)[AR] Creating static library: $(LIBBINE_CUDA)$(NC)"
	ar rcs $@ $^

$(LIB_OBJ_DIR_CUDA)/%.o: %.c
	@mkdir -p $(LIB_OBJ_DIR_CUDA)
	@echo -e "$(YELLOW)[CC] Compiling $< -> $@...$(NC)"
	$(BINECC) $(CFLAGS) -DCUDA_AWARE -ldcudart -c $< -o $@

-include $(LIB_DEPS_CUDA)

clean:
	@echo -e "[CLEAN] Removing object files and library..."
	@rm -rf $(LIB_OBJ_DIR) $(LIBBINE) $(LIB_OBJ_DIR_CUDA) $(LIBBINE_CUDA)
