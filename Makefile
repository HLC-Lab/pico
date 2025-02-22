.DEFAULT_GOAL := all

.PHONY: all clean libswing test force_rebuild

# Common settings to be shared with sub-makefiles
CFLAGS_COMMON = -Wall -I../include $(CFLAGS_COMP_SPECIFIC)

ifeq ($(DEBUG),1)
	CFLAGS_COMMON += -DDEBUG -g
endif
export CFLAGS_COMMON

# Read the previous DEBUG flag value, if available.
PREV_DEBUG := $(shell [ -f obj/.debug_flag ] && cat obj/.debug_flag)

# Build all components
all: force_rebuild libswing test

force_rebuild:
	@if [ ! -f obj/.debug_flag ] || [ "$(PREV_DEBUG)" != "$(DEBUG)" ]; then \
		echo -e "$(RED)[BUILD] DEBUG flag changed or .debug_flag missing. Cleaning subdirectories...$(NC)"; \
		$(MAKE) -C libswing clean; \
		$(MAKE) -C test clean; \
		echo "$(DEBUG)" > obj/.debug_flag; \
	else \
		echo -e "$(BLUE)[BUILD] DEBUG flag unchanged...$(NC)"; \
	fi

# Build the libswing static library
libswing:
	@echo -e "$(BLUE)[BUILD] Compiling libswing static library...$(NC)"
	$(MAKE) -C libswing $(if $(DEBUG),DEBUG=$(DEBUG))

# Build the test executable
test: libswing
	@echo -e "$(BLUE)[BUILD] Compiling test executable...$(NC)"
	$(MAKE) -C test $(if $(DEBUG),DEBUG=$(DEBUG))

# Clean all builds
clean:
	@echo -e "${RED}[CLEAN] Cleaning all builds...$(NC)"
	@$(MAKE) -C libswing clean
	@$(MAKE) -C test clean
