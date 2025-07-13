#!/bin/bash
# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT

[[ -n "$SLURM_PARAMS" ]] && inform "Sbatch params:" "$SLURM_PARAMS"

####################################################################################
#                           MAIN BENCHMARKING LOOP                                 #
####################################################################################
iter=0
for config in ${TEST_CONFIG_FILES[@]//,/ }; do
    export TEST_CONFIG=${config}
    export TEST_ENV="${TEST_CONFIG}_env.sh"

    # Now --gpu-per-node is analyzed. It is a comma separated list of GPUs per node.
    # We need to iterate through it and set the GPU_AWARENESS variable accordingly.
    # When a 0 is found, the GPU_AWARENESS is set to no and the test is run on CPU,
    # iterating through the --tasks-per-node list.
    for n_gpu in ${GPU_PER_NODE[@]//,/ }; do
        if [[ "$n_gpu" == "0" ]]; then
            export GPU_AWARENESS="no"
            export BENCH_EXEC=$BENCH_EXEC_CPU

            for ntasks in ${TASKS_PER_NODE[@]//,/ }; do
                export CURRENT_TASKS_PER_NODE=$ntasks
                export MPI_TASKS=$(( N_NODES * CURRENT_TASKS_PER_NODE ))

                # --ntasks will override any --tasks-per-node value,
                # CURRENT_TASKS_PER_NODE is set for metadata reasons
                # and is a truncated value not representative of actual allocation.
                if [[ -n "$FORCE_TASKS" ]]; then
                    export MPI_TASKS=$FORCE_TASKS
                    export CURRENT_TASKS_PER_NODE=$(( FORCE_TASKS / N_NODES ))
                fi

                # Run script to parse and generate test environment variables
                python3 $BINE_DIR/config/parse_test.py || exit 1
                source $TEST_ENV
                load_other_env_var
                success "📄 Test configuration ${TEST_CONFIG} parsed (CPU, ntasks=${CURRENT_TASKS_PER_NODE})"

                # Create the metadata if not in debug mode or dry run
                if [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" ]]; then
                    export DATA_DIR="$OUTPUT_DIR/$iter"
                    mkdir -p "$DATA_DIR"
                    python3 $BINE_DIR/results/generate_metadata.py $iter || exit 1
                    success "📂 Metadata of $DATA_DIR created"
                fi

                print_sanity_checks

                # Run the tests
                run_all_tests
                ((iter++))

                # If --ntasks is set, we skip the possible --tasks-per-node values
                if [[ -n "$FORCE_TASKS" ]]; then
                    warning "--ntasks is set, skipping possible --tasks-per-node values"
                    break
                fi
            done
        else
            export GPU_AWARENESS="yes"
            export CURRENT_TASKS_PER_NODE=$n_gpu
            export MPI_TASKS=$(( N_NODES * n_gpu ))
            export BENCH_EXEC=$BENCH_EXEC_GPU

            # Run script to parse and generate test environment variables
            python3 $BINE_DIR/config/parse_test.py || exit 1
            source $TEST_ENV
            load_other_env_var
            success "📄 Test configuration ${TEST_CONFIG} parsed (GPU, gpus per node=${CURRENT_TASKS_PER_NODE})"

            # Create the metadata if not in debug mode or dry run
            if [[ "$DEBUG_MODE" == "no" && "$DRY_RUN" == "no" ]]; then
                export DATA_DIR="$OUTPUT_DIR/$iter"
                mkdir -p "$DATA_DIR"
                python3 $BINE_DIR/results/generate_metadata.py $iter || exit 1
                success "📂 Metadata of $DATA_DIR created"
            fi

            print_sanity_checks

            # Run the tests
            run_all_tests
            ((iter++))
        fi
    done
done

success "All tests completed successfully"

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

####################################################################################
#               ANALYZE ALLOCATION AND GENERATE COMMUNICATION TRACE                #
####################################################################################
if [[ $LOCATION != "local" ]]; then
    python $BINE_DIR/tracer/trace_communications.py --alloc "$OUTPUT_DIR/alloc.csv" --location $LOCATION --save
    success "📊 Trace of communications generated"

    squeue -j $SLURM_JOB_ID
fi
