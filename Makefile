.DEFAULT_GOAL := all

.PHONY: all clean libbine pico_core force_rebuild

obj:
	@mkdir -p obj

CFLAGS_COMMON = -O3 -Wall -I$(BINE_DIR)/include -MMD -MP

ifeq ($(DEBUG),1)
	CFLAGS_COMMON += -DDEBUG -g
endif
export CFLAGS_COMMON

all: force_rebuild libbine pico_core

PREV_DEBUG := $(shell [ -f obj/.debug_flag ] && cat obj/.debug_flag)
PREV_LIB := $(shell [ -f obj/.last_lib ] && cat obj/.last_lib)
PREV_CUDA_AWARE := $(shell [ -f obj/.cuda_aware ] && cat obj/.cuda_aware)

force_rebuild: obj
	@if [[ ! -f obj/.debug_flag || ! -f obj/.last_lib || ! -f obj/.cuda_aware || "$(PREV_DEBUG)" != "$(DEBUG)" || "$(PREV_LIB)" != "$(MPI_LIB)" || "$(PREV_CUDA_AWARE)" != "$(CUDA_AWARE)" ]]; then \
		echo -e "$(RED)[BUILD] LIB, DEBUG or CUDA flag changed. Cleaning subdirectories...$(NC)"; \
		$(MAKE) -C libbine clean; \
		$(MAKE) -C pico_core clean; \
		echo "$(DEBUG)" > obj/.debug_flag; \
		echo "$(MPI_LIB)" > obj/.last_lib; \
		echo "$(CUDA_AWARE)" > obj/.cuda_aware; \
	else \
		echo -e "$(BLUE)[BUILD] LIB, DEBUG or CUDA flag unchanged...$(NC)"; \
	fi

libbine:
	@echo -e "$(BLUE)[BUILD] Compiling libbine static library...$(NC)"
	$(MAKE) -C libbine $(if $(DEBUG),DEBUG=$(DEBUG)) $(if $(CUDA_AWARE),CUDA_AWARE=$(CUDA_AWARE))

pico_core: libbine
	@echo -e "$(BLUE)[BUILD] Compiling pico_core executable...$(NC)"
	$(MAKE) -C pico_core $(if $(DEBUG),DEBUG=$(DEBUG)) $(if $(CUDA_AWARE),CUDA_AWARE=$(CUDA_AWARE))

clean:
	@echo -e "${RED}[CLEAN] Cleaning all builds...$(NC)"
	@$(MAKE) -C libbine clean
	@$(MAKE) -C pico_core clean
	@rm -rf obj
