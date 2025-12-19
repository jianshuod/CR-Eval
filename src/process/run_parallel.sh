#!/bin/bash

input_dir="/input"

# Find all files and process them in parallel with retries and timeout
find "$input_dir" -type f | parallel --timeout 300 --retries 3 -j $((4 * $(nproc))) /convert_doc.sh {}
