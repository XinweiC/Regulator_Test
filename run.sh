#!/bin/bash
TargetPower="2 5 10 12 15 17 20"
declare -a TIMES=("20000" "5000" "10000") # control cycle in micro seconds
REGULATOR="../../pr/rg" # path to the regulator
PROGRAM="runme.sh"
RESULT_DIR="../../pr/result/report" #where to store results
RESULT_PREFIX="barnes" # name of the results
PR_DIR="$PWD"
mkdir -p $RESULT_DIR
for ((i=0;i<${#TIMES[@]};++i)); do
    for TP in $TargetPower; do
       # for PROGRAM in $PROGRAMS; do
        cd ../benchmarks/barnes
 	RESULT_PREFIX=barnes 
        #COMMAND="sudo timeout 1s $REGULATOR $PROGRAM $CF ${TIMES[i]} > ${RESULT_DIR}/${DATE}${PROGRAM}${RESULT_PREFIX}_${CF}_${TIMES[i]}"
                #echo "$COMMAND"
                #echo "timeout 1 sudo $REGULATOR $PROGRAM $CF ${TIMES[i]}" 
                #timeout 1 sudo $REGULATOR $PROGRAM $CF ${TIMES[i]} > ${RESULT_DIR}/${DATE}${PROGRAM}${RESULT_PREFIX}_${CF}_${TIMES[i]}
                #timeout -k 5 1  sudo $REGULATOR $PROGRAM $CF ${TIMES[i]}
                #timeout 1s sudo $REGULATOR $PROGRAM $CF ${TIMES[i]} 
        COMMAND="sudo timeout 15s $REGULATOR $PROGRAM $TP ${TIMES[i]} > ${RESULT_DIR}${RESULT_PREFIX}_${TP}_${TIMES[i]}"

	echo "$COMMAND"
	sudo timeout 15s $REGULATOR $PROGRAM $TP ${TIMES[i]} > ${RESULT_DIR}${RESULT_PREFIX}_${TP}_${TIMES[i]}
        cd $PR_DIR
	#done
    done
done
