.DEFAULT_GOAL := all

.PHONY: all clean libswing bench force_rebuild

# Common settings to be shared with sub-makefiles
CFLAGS_COMMON = -Wall -I$(SWING_DIR)/include $(CFLAGS_COMP_SPECIFIC)

ifeq ($(DEBUG),1)
	CFLAGS_COMMON += -DDEBUG -g
endif
export CFLAGS_COMMON

# Read the previous DEBUG flag value, if available.
PREV_DEBUG := $(shell [ -f obj/.debug_flag ] && cat obj/.debug_flag)
PREV_LIB := $(shell [ -f obj/.last_lib ] && cat obj/.last_lib)

# Build all components
all: force_rebuild libswing bench

force_rebuild:
	@if [ ! -f obj/.debug_flag ] || [ ! -f obj/.last_lib ] || [ "$(PREV_DEBUG)" != "$(DEBUG)" ] || [ "$(PREV_LIB)" != "$(MPI_LIB)" ]; then \
		echo -e "$(RED)[BUILD] LIB or DEBUG flag changed. Cleaning subdirectories...$(NC)"; \
		$(MAKE) -C libswing clean; \
		$(MAKE) -C bench clean; \
		echo "$(DEBUG)" > obj/.debug_flag; \
		echo "$(MPI_LIB)" > obj/.last_lib; \
	else \
		echo -e "$(BLUE)[BUILD] LIB and DEBUG flag unchanged...$(NC)"; \
	fi

# Build the libswing static library
libswing:
	@echo -e "$(BLUE)[BUILD] Compiling libswing static library...$(NC)"
	$(MAKE) -C libswing $(if $(DEBUG),DEBUG=$(DEBUG))

# Build the bench executable
bench: libswing
	@echo -e "$(BLUE)[BUILD] Compiling bench executable...$(NC)"
	$(MAKE) -C bench $(if $(DEBUG),DEBUG=$(DEBUG))

# Clean all builds
clean:
	@echo -e "${RED}[CLEAN] Cleaning all builds...$(NC)"
	@$(MAKE) -C libswing clean
	@$(MAKE) -C bench clean
