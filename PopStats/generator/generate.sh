#!/bin/bash

mkdir -p data
for i in `seq 1 4`; do
    mkdir -p data/task$i
done

matlab -nodisplay -nodesktop -r "generate_dataset; exit"
