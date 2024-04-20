#!/bin/bash

copyright_notice=$(cat copyright.txt)

for file in $(git ls-files | grep '\.py$'); do
  # Prepend the copyright notice to the Python file
  (echo "$copyright_notice" && cat "$file") > "$file.tmp"
  mv "$file.tmp" "$file"
done
