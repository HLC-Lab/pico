#!/bin/bash
system="leonardo"
MIN_LINES=16

echo "Nodes,Reduction" > sinfo_summary_${system}.csv
total_reduction=0
count=0
for f in $(ls sinfo_${system})
do
    fname="./sinfo_${system}/$f"
    total_lines=$(wc -l < "$fname")

    if [ "$total_lines" -lt ${MIN_LINES} ]; then
        continue
    fi

    # Compute the largest power of 2 less than or equal to total_lines
    lines_to_keep=$((2**$(echo "l($total_lines)/l(2)" | bc -l | cut -d. -f1)))

    # Prepend to each line of the file $f its line number
    echo "MPI_Rank,allocation" > tmp.csv
    awk -v max="$lines_to_keep" 'NR <= max {print NR "," $0}' "$fname" >> tmp.csv

    python3 trace_communications.py --location leonardo --alloc tmp.csv --comm algo_patterns.json --map maps/${system}.txt --save --out out.csv &>/dev/null
    BINE_BDW_ALLREDUCE=$(cat out.csv | grep allreduce | grep swing_bandwidth | cut -d ',' -f4)
    BINOMIAL_BDW_ALLREDUCE=$(cat out.csv | grep allreduce | grep rabenseifner | cut -d ',' -f4)
    #BINE_BDW_ALLREDUCE=$(cat out.csv | grep bcast | grep swing_halving | cut -d ',' -f4)
    #BINOMIAL_BDW_ALLREDUCE=$(cat out.csv | grep bcast | grep binomial_halving | cut -d ',' -f4)
    # Print the ratio
    REDUCTION=$(echo "scale=2; ( $BINOMIAL_BDW_ALLREDUCE - $BINE_BDW_ALLREDUCE ) * 100.0 / $BINOMIAL_BDW_ALLREDUCE" | bc)
    total_reduction=$(echo "$total_reduction + $REDUCTION" | bc)
    count=$((count + 1))
    echo "$lines_to_keep,$REDUCTION" >> sinfo_summary_${system}.csv
    rm tmp.csv
done

average=$(echo "scale=2; $total_reduction / $count" | bc)
echo "Average REDUCTION on > $MIN_LINES nodes: $average%"