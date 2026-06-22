# Makefile for the TWILL helper utilities.
#
# test-twill: build & run the standalone TWILL schedule unit test. It is a single
# process (no mpirun) and makes no MPI calls; it is compiled with mpicc only
# because twill_schedule_test.c includes ../libpico/libpico_utils.h, which pulls
# in mpi.h for its other helpers.
#
#   make -C utils test-twill        # or, from this folder:  make test-twill

TWILL_TEST_CC  ?= mpicc
TWILL_TEST_BIN ?= twill_schedule_test

.DEFAULT_GOAL := test-twill
.PHONY: test-twill clean

test-twill:
	@echo "[TEST] Building & running TWILL schedule unit test (single process, no mpirun)..."
	$(TWILL_TEST_CC) -O2 -Wall -I../libpico twill_schedule_test.c -o $(TWILL_TEST_BIN)
	@./$(TWILL_TEST_BIN)

clean:
	@rm -f $(TWILL_TEST_BIN)
