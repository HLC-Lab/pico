#!/bin/bash

# Colors for styling output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Print error messages in red
error() {
    echo -e "\n${RED}❌❌❌ ERROR: $1 ❌❌❌${NC}\n" >&2
}

# Print success messages in green
success() {
    echo -e "\n${GREEN}$1${NC}\n"
}

# Cleanup function for SIGINT
cleanup() {
    error "Caught Ctrl+C! Killing all child processes..."
    pkill -P $$
    exit 1
}

# Source the environment configuration
source_environment() {
    local env_file="scripts/environments/$1.sh"
    if [ -f "$env_file" ]; then
        source "$env_file"
        return 0
    else
        return 1
    fi
}


# Compile the codebase
compile_code() {
    make clean
    make_command="make all"
    [ "$ENABLE_OMPI_TEST" == "yes" ] && make_command="$make_command OMPI_TEST=1"
    [ "$DEBUG_MODE" == "yes" ] && make_command="$make_command DEBUG=1"

    if ! $make_command; then
        error "Compilation failed. Exiting."
        return 1
    fi

    success "Compilation succeeded."
    return 0
}

# Determine the number of iterations based on array size
get_iterations() {
    local size=$1
    if [ "$DEBUG_MODE" == "yes" ]; then
        echo 1
    elif [ $size -le 512 ]; then
        echo 20000
    elif [ $size -le 1048576 ]; then
        echo 2000
    elif [ $size -le 8388608 ]; then
        echo 200
    elif [ $size -le 67108864 ]; then
        echo 20
    else
        echo 5
    fi
}

select_algorithms() {
    local ALGO_JSON="$1"
    local TEST_JSON="$2"

    # Ensure ALGO_JSON is provided
    if [[ -z "$ALGO_JSON" ]] || [[ -z "$TEST_JSON" ]]; then
        error "ALGO_JSON or TEST_JSON is not set."
        return 1
    fi

    # Read test config parameters
    local COLLECTIVE=$(jq -r '.collective_type' "$TEST_JSON")
    local INCLUDE_TAGS=($(jq -r '.include_by_tags[]' "$TEST_JSON"))
    local EXCLUDE_TAGS=($(jq -r '.exclude_by_tags[]' "$TEST_JSON"))
    local INCLUDE_SPECIFIC=($(jq -r '.include_specific_algorithms[]' "$TEST_JSON"))
    local EXCLUDE_SPECIFIC=($(jq -r '.exclude_specific_algorithms[]' "$TEST_JSON"))

    # Check if jq is installed
    if ! command -v jq &> /dev/null; then
        error "jq is not installed. Please install jq to proceed."
        return 1
    fi

    # Check if COLLECTIVE_TYPE exists in ALGO_JSON
    if ! jq -e "has(\"$COLLECTIVE\")" "$ALGO_JSON" &>/dev/null; then
        error "Collective type '$COLLECTIVE' not found in $ALGO_JSON."
        return 1
    fi

    echo "Include tags: ${INCLUDE_TAGS[@]}"
    # Step 1: Include algorithms by tag
    declare -A SELECTED_ALGOS
    while read -r algo_id; do
        TAGS=($(jq -r ".\"$COLLECTIVE\".\"$algo_id\".tags[]" "$ALGO_JSON" 2>/dev/null))

        for tag in "${INCLUDE_TAGS[@]}"; do
            if [[ " ${TAGS[*]} " =~ " $tag " ]]; then
                SELECTED_ALGOS["$algo_id"]=1
                break
            fi
        done
    done < <(jq -r "keys[]" <<<"$(jq -r ".\"$COLLECTIVE\"" "$ALGO_JSON")")

    # Step 2: Exclude algorithms by tag
    echo "Exclude tags: ${EXCLUDE_TAGS[@]}"
    for algo in "${!SELECTED_ALGOS[@]}"; do
        TAGS=($(jq -r ".\"$COLLECTIVE\".\"$algo\".tags[]" "$ALGO_JSON" 2>/dev/null))

        for tag in "${EXCLUDE_TAGS[@]}"; do
            if [[ " ${TAGS[*]} " =~ " $tag " ]]; then
                unset "SELECTED_ALGOS[$algo]"
                break
            fi
        done
    done

    # Step 3: Include specific algorithms
    echo "Include specific: ${INCLUDE_SPECIFIC[@]}"
    for algo in "${INCLUDE_SPECIFIC[@]}"; do
        SELECTED_ALGOS["$algo"]=1
    done

    # Step 4: Exclude specific algorithms
    echo "Exclude specific: ${EXCLUDE_SPECIFIC[@]}"
    for algo in "${EXCLUDE_SPECIFIC[@]}"; do
        unset "SELECTED_ALGOS[$algo]"
    done

    # Step 5: Preserve order while sorting numerically
    local FINAL_ALGOS=($(printf "%s\n" "${!SELECTED_ALGOS[@]}" | sort -n))

    # Check if the final algorithm list is empty
    if [[ ${#FINAL_ALGOS[@]} -eq 0 ]]; then
        error "No algorithms selected after applying all filters."
        return 1
    fi

    # Export ALGOS and COLLECTIVE_TYPE
    export ALGOS="${FINAL_ALGOS[@]}"
    export COLLECTIVE_TYPE="$COLLECTIVE"

    success "Algorithm successfully selected: $ALGOS"
    return 0
}

get_algorithm_names() {
    local ALGO_JSON="$1"

    if [[ -z "$ALGO_JSON" ]] || [[ -z "$COLLECTIVE_TYPE" ]] || [[ -z "$ALGOS" ]]; then
        error "ALGO_JSON, COLLECTIVE_TYPE, or ALGOS is not set."
        return 1
    fi

    local ALGO_NAMES=()

    for algo_id in $ALGOS; do
        local name
        name=$(jq -r ".\"$COLLECTIVE_TYPE\".\"$algo_id\".name" "$ALGO_JSON" 2>/dev/null)

        if [[ -n "$name" && "$name" != "null" ]]; then
            ALGO_NAMES+=("$name")
        else
            ALGO_NAMES+=("UNKNOWN_ALGO_$algo_id")  # Fallback if algo ID is not found
        fi
    done

    export NAMES="${ALGO_NAMES[@]}"

    return 0

}

get_algorithm_by_tag() {
    local ALGO_JSON="$1"
    local VAR="$2"
    local TAG="$3"

    if [[ -z "$ALGO_JSON" ]] || [[ -z "$COLLECTIVE_TYPE" ]] || [[ -z "$ALGOS" ]] || [[ -z "$TAG" ]] || [[ -z "$VAR" ]]; then
        error "ALGO_JSON, COLLECTIVE_TYPE, ALGOS, TAG, or VAR is not set."
        return 1
    fi

    local TAG_ALGOS=()

    for algo_id in $ALGOS; do
        local tags
        tags=$(jq -r ".\"$COLLECTIVE_TYPE\".\"$algo_id\".tags | join(\" \")" "$ALGO_JSON" 2>/dev/null)

        if [[ " $tags " =~ " $TAG " ]]; then
            TAG_ALGOS+=("$algo_id")
        fi
    done

    eval "$VAR=\"${TAG_ALGOS[@]}\""

    return 0
}
# get_algorithm_skips() {
#     local ALGO_JSON="$1"
#
#     if [[ -z "$ALGO_JSON" ]] || [[ -z "$COLLECTIVE_TYPE" ]] || [[ -z "$ALGOS" ]]; then
#         error "ALGO_JSON, COLLECTIVE_TYPE, or ALGOS is not set."
#         return 1
#     fi
#
#     local SKIP_ALGOS=()
#
#     # Loop through all selected algorithms in ALGOS
#     for algo_id in $ALGOS; do
#         local tags
#         tags=$(jq -r ".\"$COLLECTIVE_TYPE\".\"$algo_id\".tags | join(\" \")" "$ALGO_JSON" 2>/dev/null)
#
#         if [[ " $tags " =~ " skip " ]]; then
#             echo "Skipping algorithm $algo_id"
#             SKIP_ALGOS+=("$algo_id")
#         fi
#     done
#
#     # Export SKIP variable with selected algorithms having the 'skip' tag
#     export SKIP="${SKIP_ALGOS[@]}"
#
#     return 0
# }

# Function to run a single test case
# Arguments: array size, iterations, data type, algorithm index
run_test() {
    local size=$1
    local iter=$2
    local type=$3
    local algo=$4
    local debug_mode=$5
    local algo_name=$6

    if [ "$debug_mode" == "yes" ]; then
      echo "DEBUG: $COLLECTIVE_TYPE -> $N_NODES processes, $size array size, $type datatype ($algo: $algo_name)"
    else
      echo "Benchmarking $COLLECTIVE_TYPE -> $N_NODES processes, $size array size, $type datatype ($algo: $algo_name Iter: $iter)"
    fi

    $RUN $RUNFLAGS -n $N_NODES $TEST_EXEC $size $iter $type $algo $OUTPUT_DIR
}

# Test algorithms here, loop through:
# - algorithms
# - number of mpi processes
# - size of the array in number of elements (of type = TYPE) 
#
# note that, if an algorithm is not internal to Open MPI, the dynamic
# rule file will be set to 0 (i.e. automatic default algorithm selection)
run_all_tests() {
    local nodes=$1
    local algos=($2)
    local sizes=($3)
    local types=($4)
    local output_dir=$5
    local debug_mode=$6
    local names=($7)

    for algo in ${algos[@]}; do
        # Update dynamic rule file for the algorithm
        $RULE_UPDATER_EXEC $RULE_FILE_PATH $algo
        export OMPI_MCA_coll_tuned_dynamic_rules_filename=${RULE_FILE_PATH}

        for size in ${sizes[@]}; do
            # Skip specific algorithms if conditions are met
            if [[ size -lt $nodes && " ${COLLECTIVE_SKIPS[$COLLECTIVE_TYPE]} " =~ " ${algo} " ]]; then
                echo "Skipping algorithm $algo for size=$size < N_NODES=$nodes"
                continue
            fi

            # Get the number of iterations for the size
            local iter=$(get_iterations $size)
            for type in ${types[@]}; do
                # Run the test for the given configuration
                run_test $size $iter $type $algo $debug_mode ${names[$algo]}
            done
        done
    done
}
