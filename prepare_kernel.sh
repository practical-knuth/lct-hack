#!/bin/bash

file="./2. ГПУ/"

if [ -e "$file" ]; then
    mkdir ./data
    cp -r "$file" ./data/
else 
    echo "File $file does not exist"
fi

bash create_kernel.sh
