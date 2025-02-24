#!/bin/bash
# Trap SIGINT (Ctrl+C) and call cleanup function
trap cleanup SIGINT

# Validate and initialize N_NODES
N_NODES=$1
if [[ -z "$N_NODES" ]] || [[ ! "$N_NODES" =~ ^[0-9]+$ ]] || [ "$N_NODES" -lt 2 ]; then
    error "N_NODES is not given or not set correctly."
    exit 1
fi

IFS=' ' read -r -a TEST_CONFIG_FILES <<< "$TEST_CONFIG_FILES"

for i in ${!TEST_CONFIG_FILES[@]}; do
    ###################################################################################
    #               PARSE THE TEST CONFIGURATION FILE TO GET THE TEST VARIABLES       #
    ###################################################################################
    export TEST_CONFIG=${TEST_CONFIG_FILES[$i]}
    export TEST_ENV="${TEST_CONFIG}_env.sh"
    python3 $SWING_DIR/config/parse_test.py || exit 1
    source $TEST_ENV
    load_other_env_var # Load env var dependant on test/environment combination
    success "📄 Test configuration ${TEST_CONFIG} parsed"
    ###################################################################################
    #               CREATE OUTPUT DIRECTORY AND GENERATE METADATA                     #
    #               ALTERNATIVELY, USE DEBUG VARIABLES                                #
    ###################################################################################
    if [ $DEBUG_MODE == "no" ]; then
        export DATA_DIR="$OUTPUT_DIR/$i"
        mkdir -p "$DATA_DIR"
        # Generate test metadata
        python3 $SWING_DIR/results/generate_metadata.py $i || exit 1
        success "📂 Metadata of $DATA_DIR created"
    else
        # export COLLECTIVE_TYPE="ALLREDUCE"
        # export ALGOS="swing_bdw_static_over"
        export ARR_SIZES="2048"
        export TYPES="int32" # For now only int,int32,int64 are supported in debug mode 
    fi
        export ARR_SIZES="2048"
        export TYPES="int32" # For now only int,int32,int64 are supported in debug mode 

    # Sanity checks
    success "==========================================================\n\t\t SANITY CHECKS"
    echo "Running test configuration: ${TEST_CONFIG_FILES[$i]}"
    echo "Running tests in: $LOCATION"
    echo "Debug mode: $DEBUG_MODE"
    echo "Number of nodes: $N_NODES"
    echo "Saving results in: $DATA_DIR"
    echo "Running benchmarks for collective: $COLLECTIVE_TYPE"
    echo -e "For algorithms: \n $ALGOS"
    echo -e "With sizes: \n $ARR_SIZES"
    echo -e "And data types: \n $TYPES"
    echo "MPI Library: $MPI_LIB, $MPI_LIB_VERSION"
    echo "Libswing Version: $LIBSWING_VERSION"
    echo "CUDA Enabled: $CUDA"
    echo "NOTES: $NOTES"
    success "=========================================================="

    ###################################################################################
    #               RUN THE TESTS FOR THE GIVEN CONFIGURATION                         #
    ###################################################################################
    run_all_tests || exit 1
done

###################################################################################
#              COMPRESS THE RESULTS AND ADD OUTPUT_DIR TO GITIGNORE               #
###################################################################################
if [[ $DEBUG_MODE == "no" ]] && [[ $LOCATION != "local" ]]; then
    $SWING_DIR/results/compress_results.sh
fi
