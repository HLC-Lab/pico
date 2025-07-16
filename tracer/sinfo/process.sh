#!/bin/bash
COLLECTIVE="allreduce"
ALGO_BINOMIAL="rabenseifner"
ALGO_BINE="bine_bandwidth"
#ALGO_BINOMIAL="recursive_doubling"
#ALGO_BINE="bine_latency"

# Collective to uppercase
COLLECTIVE_UPPER=$(echo "$COLLECTIVE" | tr '[:lower:]' '[:upper:]')
for system in "leonardo" "lumi"
do
    echo "Nodes,Groups,Reduction" > ${system}_${COLLECTIVE}_${ALGO_BINOMIAL}_vs_${ALGO_BINE}.csv
    total_reduction=0
    count=0
    for f in $(ls ${system})
    do
        #echo $f
        fname="./${system}/$f"

        # if 'viz' in the content of the file, skip it
        if grep -q "viz" "$fname"; then
            continue
        fi

        total_lines=$(wc -l < "$fname")

        if [ "$total_lines" -lt 2 ]; then
            continue
        fi

        # Compute the largest power of 2 less than or equal to total_lines
        #lines_to_keep=$((2**$(echo "l($total_lines)/l(2)" | bc -l | cut -d. -f1)))
        lines_to_keep=$total_lines

        # Prepend to each line of the file $f its line number
        echo "MPI_Rank,allocation" > tmp.csv
        awk -v max="$lines_to_keep" 'NR <= max {print (NR-1) "," $0}' "$fname" >> tmp.csv

        #echo "Processing $fname with $lines_to_keep lines"
        #echo "python3 ../trace_communications.py --location ${system} --alloc tmp.csv --comm ../algo_patterns.json --map ../maps/${system}.txt --save --out out.csv --hostname_only --coll ${COLLECTIVE_UPPER}"
        
        OUT=$(python3 ../trace_communications.py --location ${system} --alloc tmp.csv --comm ../algo_patterns.json --map ../maps/${system}.txt --save --out out.csv --hostname_only --coll ${COLLECTIVE_UPPER})
        NUM_GROUPS=$(echo "$OUT" | grep "Num Cells" | cut -d ':' -f 2 | cut -d ' ' -f 2)
        #exit 

        # If NUM_GROUPS is 1, skip the file
        if [ "$NUM_GROUPS" -eq 1 ]; then
            rm tmp.csv
            rm -f out.csv
            continue
        fi
        
        if [ -f "out.csv" ]; then
            BINE_BYTES=$(cat out.csv | grep ${COLLECTIVE} | grep ${ALGO_BINE} | cut -d ',' -f4)
            BINOMIAL_BYTES=$(cat out.csv | grep ${COLLECTIVE} | grep ${ALGO_BINOMIAL} | cut -d ',' -f4)
            # Print the ratio (if BINOMIAL_BYTES is not zero)
            if [ "$BINOMIAL_BYTES" != "0" ]; then
                REDUCTION=$(echo "scale=2; ( $BINOMIAL_BYTES - $BINE_BYTES ) * 100.0 / $BINOMIAL_BYTES" | bc)
                total_reduction=$(echo "$total_reduction + $REDUCTION" | bc)
                count=$((count + 1))
                echo "$lines_to_keep,$NUM_GROUPS,$REDUCTION" >> ${system}_${COLLECTIVE}_${ALGO_BINOMIAL}_vs_${ALGO_BINE}.csv
            fi
            rm out.csv            
        fi
        rm tmp.csv
    done

    average=$(echo "scale=2; $total_reduction / $count" | bc)
    echo "Average REDUCTION: $average%"
done
