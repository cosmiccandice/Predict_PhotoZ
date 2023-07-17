#!/bin/bash

if [ -r ${1} ]; then

    # Location of file containing the header
    HEADER_CSV=/projects/b1094/stroh/projects/panstarrs_photoz/header.csv

    FILENAME=${1%.gz}

    # Decompress the file
    gunzip -c $1 > $FILENAME
    
    # Append header
    W_HEADER=${FILENAME%.csv}.h.csv
    cat $HEADER_CSV $FILENAME > $W_HEADER

   # Remove ,-999.0 instances
    sed -i 's/,-999\.0/,/g' $W_HEADER
    
    # Remove ,-999, instances
    sed -i 's/,-999/,/g' $W_HEADER

    rm $FILENAME

fi
