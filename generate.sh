#!/bin/bash

NUM_JOBS=2
TOTAL_JOBS=2
machine_id=1
job_count=0

for ((i=1; i<=TOTAL_JOBS; i++))
do
    echo "Run $i:"
    python ./generate.py --base_id=-1 --machine_id="machine_$machine_id" &
    ((job_count++))

    if [ "$job_count" -ge "$NUM_JOBS" ]; then
        wait
        job_count=0
    fi
done
wait

# python ./generate.py --base_id=-1 --machine_id="machine_1"
