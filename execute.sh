#!/bin/bash
# Put your command below to execute your program.
# Replace "./my-program" with the command that can execute your program.
# Remember to preserve " $@" at the end, which will be the program options we give you.

# Usage:
# ./execute.sh [-r] -i query-file -o ranked-list -m model-dir -d NTCIR-dir

# Make a directory for model files
if [[ ! -d b06902029 ]]; then 
    mkdir b06902029
    echo 'Create a directory `b06902029/` for model files'
fi

python3 main.py $@ \
                --preprocess \
                --tfidf \
                -c b06902029 \
                --model-name OkapiBM25-0xff

echo 'Finish All. Remember to remove b06902029'

