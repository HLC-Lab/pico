#!/bin/bash

source scripts/utils.sh

# Ensure that LOCATION is set
if [ -z "${LOCATION:-}" ]; then
    error "LOCATION variable is not set."
    return 1
fi

BASE_DIR="results/$LOCATION"

# Verify that BASE_DIR exists and is a directory
if [ ! -d "$BASE_DIR" ]; then
    error "'$BASE_DIR' is not a valid directory."
    return 1
fi

# Function to compress a subdirectory
compress_directory() {
    local dir="$1"
    local parent_dir archive_name

    parent_dir=$(dirname "$dir")
    archive_name="${dir}.tar.gz"

    echo "Compressing '$dir' into '$archive_name'..."
    if ! tar -czf "$archive_name" -C "$parent_dir" "$(basename "$dir")" ; then
        error "Failed to compress '$dir'."
    fi
}


# Navigate through each subdirectory in BASE_DIR
for subdir in "$BASE_DIR"/*; do
    if [ -d "$subdir" ]; then
        archive_file="${subdir}.tar.gz"
        if [ ! -f "$archive_file" ]; then
            compress_directory "$subdir"
            git add "$archive_file"
            echo "$subdir/" >> .gitignore

            success "Archive '$archive_file' staged for commit and '$subdir/' added to .gitignore."
        fi
    fi
done

# Stage .gitignore
git add .gitignore
success ".gitignore staged for commit."
