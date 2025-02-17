.PHONY: all clean libswing test

# Common settings to be shared with sub-makefiles
CFLAGS_COMMON = -Wall -I../include

# Add conditional flags
ifneq ($(filter clang gcc mpicc,$(CC)),)
	CFLAGS_COMMON += -O3 -MMD -MP
endif
ifneq ($(findstring cc craycc,$(CC)),)
	CFLAGS_COMMON += -Ofast -funroll-loops -em
endif
export CFLAGS_COMMON

# Variables for colors
BLUE := \033[1;34m
RED := \033[1;31m
NC := \033[0m

# Build all components
all: libswing test

# Build the libswing static library
libswing:
	@echo -e "$(BLUE)[BUILD] Compiling libswing static library...$(NC)"
	$(MAKE) -C libswing

# Build the test executable
test: libswing
	@echo -e "$(BLUE)[BUILD] Compiling test executable...$(NC)"
	$(MAKE) -C test $(if $(DEBUG),DEBUG=$(DEBUG))

# Clean all builds
clean:
	@echo -e "${RED}[CLEAN] Cleaning all builds...$(NC)"
	@$(MAKE) -C libswing clean
	@$(MAKE) -C test clean
