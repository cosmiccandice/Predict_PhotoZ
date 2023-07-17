#!/usr/bin/bash

if [ -r ${1} ]; then

    # Location of file containing the header
    HEADER_CSV=/projects/b1094/software/catalogs/ps1dr2/header.csv

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

    # Now only cut the columns wanted for photoz
    DB=${W_HEADER%.csv}.db.csv
    cut -d, -f 1,2,3,8,9,10,11,12,16,17,18,19,20,21,23,24,33,34,37,38,55,56,58,83,84,87,88,105,106,108,133,134,137,138,155,156,158,183,184,187,188,205,206,208,233,234,237,238,255,256,258 $W_HEADER > $DB
    
    rm $FILENAME
    rm $W_HEADER

fi
