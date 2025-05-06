#!/bin/bash
COLLECTIVE="allreduce"
ALGO_BINOMIAL="rabenseifner"
ALGO_BINE="swing_bandwidth"

for system in "leonardo" "lumi"
do
    echo "Nodes,Reduction" > sinfo_summary_${system}.csv
    total_reduction=0
    count=0
    for f in $(ls sinfo_${system})
    do
        #echo $f
        fname="./sinfo_${system}/$f"
        total_lines=$(wc -l < "$fname")

        if [ "$total_lines" -lt 2 ]; then
            continue
        fi

        # Compute the largest power of 2 less than or equal to total_lines
        lines_to_keep=$((2**$(echo "l($total_lines)/l(2)" | bc -l | cut -d. -f1)))

        # Prepend to each line of the file $f its line number
        echo "MPI_Rank,allocation" > tmp.csv
        awk -v max="$lines_to_keep" 'NR <= max {print NR "," $0}' "$fname" >> tmp.csv

        python3 trace_communications.py --location ${system} --alloc tmp.csv --comm algo_patterns.json --map maps/${system}.txt --save --out out.csv --hostname_only &>/dev/null
        
        if [ -f "out.csv" ]; then
            BINE_BYTES=$(cat out.csv | grep ${COLLECTIVE} | grep ${ALGO_BINE} | cut -d ',' -f4)
            BINOMIAL_BYTES=$(cat out.csv | grep ${COLLECTIVE} | grep ${ALGO_BINOMIAL} | cut -d ',' -f4)
            # Print the ratio
            REDUCTION=$(echo "scale=2; ( $BINOMIAL_BYTES - $BINE_BYTES ) * 100.0 / $BINOMIAL_BYTES" | bc)
            total_reduction=$(echo "$total_reduction + $REDUCTION" | bc)
            count=$((count + 1))
            echo "$lines_to_keep,$REDUCTION" >> sinfo_summary_${system}.csv
            rm out.csv            
        fi
        rm tmp.csv
    done

    average=$(echo "scale=2; $total_reduction / $count" | bc)
    echo "Average REDUCTION: $average%"
done