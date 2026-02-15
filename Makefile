.DEFAULT_GOAL := all

.PHONY: all clean libpico pico_core

CFLAGS_COMMON = -O3 -Wall -I$(PICO_DIR)/include -MMD -MP

ifeq ($(DEBUG),1)
	CFLAGS_COMMON += -DDEBUG -g
endif

ifeq ($(PICO_INSTRUMENT),1)
	CFLAGS_COMMON += -DPICO_INSTRUMENT
endif

ifeq ($(PICO_NCCL),1)
    export NCCL_FLAG = -DPICO_NCCL -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include -lmpi -lnccl
    export LDLIBS += -L$(NCCL_HOME)/lib -lnccl -L$(CUDA_HOME)/lib64 -lcudart
endif

export CFLAGS_COMMON

all: libpico pico_core

libpico:
	@echo -e "$(BLUE)[BUILD] Compiling libpico static library...$(NC)"
	$(MAKE) -C libpico $(if $(DEBUG),DEBUG=$(DEBUG)) $(if $(PICO_MPI_CUDA_AWARE),PICO_MPI_CUDA_AWARE=$(PICO_MPI_CUDA_AWARE)) $(if $(GPU_NATIV_SUPPORT),GPU_NATIV_SUPPORT=$(GPU_NATIV_SUPPORT)) $(if $(PICO_NCCL), PICO_NCCL=$(PICO_NCCL))

pico_core: libpico
	@echo -e "$(BLUE)[BUILD] Compiling pico_core executable...$(NC)"
	$(MAKE) -C pico_core $(if $(DEBUG),DEBUG=$(DEBUG)) $(if $(PICO_MPI_CUDA_AWARE),PICO_MPI_CUDA_AWARE=$(PICO_MPI_CUDA_AWARE)) $(if $(GPU_NATIV_SUPPORT),GPU_NATIV_SUPPORT=$(GPU_NATIV_SUPPORT)) $(if $(PICO_NCCL),PICO_NCCL=$(PICO_NCCL))

clean:
	@echo -e "${RED}[CLEAN] Cleaning all builds...$(NC)"
	@rm -rf bin/ obj/ lib/