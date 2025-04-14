#!/bin/bash

# Iterate over all matching directories
for dir in retro_*/*/; do
    # Extract the parent directory and subdirectory names
    parent_dir=$(dirname "$dir")
    sub_dir=$(basename "$dir")

    # Check if the subdirectory name matches the parent directory name
    if [[ "$parent_dir" == *"$sub_dir" ]]; then
        echo "Moving contents of $dir to $parent_dir and removing $dir"

        # Move contents to the parent directory
        mv "$dir"* "$parent_dir"/

        # Remove the now-empty subdirectory
        rmdir "$dir"
    fi
done
