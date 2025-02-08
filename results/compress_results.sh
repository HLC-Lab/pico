#!/bin/bash

# Base directory containing the results
BASE_DIR="./results"

# Function to compress a subdirectory
compress_directory() {
    local dir="$1"
    local parent_dir
    parent_dir=$(dirname "$dir")
    local archive_name="$dir.tar.gz"

    echo "Compressing '$dir' into '$archive_name'..."
    tar -czf "$archive_name" -C "$parent_dir" "$(basename "$dir")"
    if [ $? -eq 0 ]; then
        echo "Successfully compressed '$dir'."
	git add $archive_name

        # Add the directory to .gitignore if not already present
        if ! grep -qx "$dir/" .gitignore; then
            echo "$dir/" >> .gitignore
            echo "Added '$dir/' to .gitignore."
        else
            echo "'$dir/' is already in .gitignore, skipping addition."
        fi
    else
        echo "Failed to compress '$dir'."
    fi
}

# Process all subdirectories inside the results subdirectories
process_results_subdirectories() {
    EXCLUDE_PATTERN="local"
    for subdir in "$BASE_DIR"/*/*; do
        if [ -d "$subdir" ] && \
           [[ "$subdir" != *"/$EXCLUDE_PATTERN/"* ]] && \
           [ ! -f "${subdir}.tar.gz" ]; then
            compress_directory "$subdir"
        fi
    done
}

# Main execution
process_results_subdirectories

# Deduplicate and sort .gitignore
sort -u .gitignore -o .gitignore
echo ".gitignore cleaned and sorted."

# Stage .gitignore
git add .gitignore
echo ".gitignore staged for commit."
