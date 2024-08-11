#!/bin/bash

cd compiled

for file in *.out; do
  for i in {1..3}; do
    X=$(./$file)
    echo -e "$file\t$X"
  done
done
