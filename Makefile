.PHONY: all clean libswing test ompi_rules

# Variables for colors
BLUE := \033[1;34m
RED := \033[1;31m
NC := \033[0m

# Build all components
all: libswing test ompi_rules

# Build the libswing static library
libswing:
	@echo -e "$(BLUE)[BUILD] Compiling libswing static library...$(NC)"
	$(MAKE) -C libswing

# Build the test executable
test: libswing
	@echo -e "$(BLUE)[BUILD] Compiling test executable...$(NC)"
	$(MAKE) -C test $(if $(OMPI_TEST),OMPI_TEST=$(OMPI_TEST)) $(if $(DEBUG),DEBUG=$(DEBUG))

# Build the change_collective_rules program
ompi_rules:
	@echo -e "$(BLUE)[BUILD] Compiling ompi_rules executable...$(NC)"
	$(MAKE) -C ompi_rules $(if $(OMPI_TEST),OMPI_TEST=$(OMPI_TEST))

# Clean all builds
clean:
	@echo -e "${RED}[CLEAN] Cleaning all builds...$(NC)"
	@$(MAKE) -C libswing clean
	@$(MAKE) -C test clean
	@$(MAKE) -C ompi_rules clean
