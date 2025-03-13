.DEFAULT_GOAL := all

.PHONY: all clean libswing bench force_rebuild

obj:
	@mkdir -p obj

# Common settings to be shared with sub-makefiles
CFLAGS_COMMON = -O3 -Wall -I$(SWING_DIR)/include -MMD -MP

ifeq ($(DEBUG),1)
	CFLAGS_COMMON += -DDEBUG -g
endif

ifeq ($(CUDA_AWARE),1)
	CFLAGS_COMMON += -DCUDA_AWARE -lcudart
endif
export CFLAGS_COMMON

# Read the previous DEBUG flag value, if available.
PREV_DEBUG := $(shell [ -f obj/.debug_flag ] && cat obj/.debug_flag)
PREV_LIB := $(shell [ -f obj/.last_lib ] && cat obj/.last_lib)
PREV_CUDA_AWARE := $(shell [ -f obj/.cuda_aware ] && cat obj/.cuda_aware)

# Build all components
all: force_rebuild libswing bench

force_rebuild: obj
	@if [ ! -f obj/.debug_flag ] || [ ! -f obj/.last_lib ] || [ "$(PREV_DEBUG)" != "$(DEBUG)" ] || [ "$(PREV_LIB)" != "$(MPI_LIB)" ] || [ "$(PREV_CUDA_AWARE)" != "$(CUDA_AWARE)" ]; then \
		echo -e "$(RED)[BUILD] LIB, DEBUG or CUDA flag changed. Cleaning subdirectories...$(NC)"; \
		$(MAKE) -C libswing clean; \
		$(MAKE) -C bench clean; \
		echo "$(DEBUG)" > obj/.debug_flag; \
		echo "$(MPI_LIB)" > obj/.last_lib; \
		echo "$(CUDA_AWARE)" > obj/.cuda_aware; \
	else \
		echo -e "$(BLUE)[BUILD] LIB, DEBUG or CUDA flag unchanged...$(NC)"; \
	fi

# Build the libswing static library
libswing:
	@echo -e "$(BLUE)[BUILD] Compiling libswing static library...$(NC)"
	$(MAKE) -C libswing $(if $(DEBUG),DEBUG=$(DEBUG)) $(if $(CUDA_AWARE),CUDA_AWARE=$(CUDA_AWARE))

# Build the bench executable
bench: libswing
	@echo -e "$(BLUE)[BUILD] Compiling bench executable...$(NC)"
	$(MAKE) -C bench $(if $(DEBUG),DEBUG=$(DEBUG)) $(if $(CUDA_AWARE),CUDA_AWARE=$(CUDA_AWARE))

# Clean all builds
clean:
	@echo -e "${RED}[CLEAN] Cleaning all builds...$(NC)"
	@$(MAKE) -C libswing clean
	@$(MAKE) -C bench clean
	@rm -f obj/.debug_flag obj/.last_lib obj/.cuda_aware
