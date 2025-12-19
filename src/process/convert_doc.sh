#!/bin/bash

file="$1"
output_dir="/output"
filename=$(basename -- "$file")
extension="${filename##*.}"
filename="${filename%.*}"

output_file="${output_dir}/${filename}.docx"

if [ -f "$output_file" ]; then
  echo "Output file ${output_file} already exists. Skipping."
  exit 0
fi

if [ "$extension" = "doc" ]; then
  # Convert .doc to .docx
  libreoffice --headless --convert-to docx "$file" --outdir "$output_dir"
elif [ "$extension" = "docx" ]; then
  # Copy .docx files directly
  cp "$file" "$output_file"
fi
