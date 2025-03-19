#!/bin/bash
# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT


iter=0
for config in ${TEST_CONFIG_FILES[@]//,/ }; do
    for gpu in ${GPU_PER_NODE[@]//,/ }; do
        if [[ "$gpu" == "0" ]]; then
            export CUDA="False"
            export MPI_TASKS=$N_NODES
            export CURRENT_TASK_PER_NODE=1
            export BENCH_EXEC=$BENCH_EXEC_CPU
        else
            export CUDA="True"
            export MPI_TASKS=$(expr $N_NODES \* $gpu)
            export CURRENT_TASK_PER_NODE=$gpu
            export BENCH_EXEC=$BENCH_EXEC_CUDA
        fi
        export TEST_CONFIG=${config}
        export TEST_ENV="${TEST_CONFIG}_env.sh"
        python3 $SWING_DIR/config/parse_test.py || exit 1
        source $TEST_ENV
        load_other_env_var # Load env var dependant on test/environment combination
        success "📄 Test configuration ${TEST_CONFIG} parsed"

        if [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" ]]; then
            export DATA_DIR="$OUTPUT_DIR/$iter"
            mkdir -p "$DATA_DIR"
            python3 $SWING_DIR/results/generate_metadata.py $iter || exit 1
            success "📂 Metadata of $DATA_DIR created"
        fi

        print_sanity_checks

        run_all_tests
        ((iter++))
    done
done

success "All tests completed successfully"

if [[ $LOCATION != "local" ]]; then
    squeue -j $SLURM_JOB_ID
fi

###################################################################################
#              COMPRESS THE RESULTS AND DELETE THE OUTPUT DIR IF REQUESTED        #
###################################################################################
if [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" && "$COMPRESS" == "yes" ]]; then
    tarball_path="$(dirname "$OUTPUT_DIR")/$(basename "$OUTPUT_DIR").tar.gz"
    if tar -czf "$tarball_path" -C "$(dirname "$OUTPUT_DIR")" "$(basename "$OUTPUT_DIR")"; then
        if [[ "$DELETE" == "yes" ]]; then
            rm -rf "$OUTPUT_DIR"
        fi
    fi
fi
