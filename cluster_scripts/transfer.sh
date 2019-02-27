#!/bin/bash
var=1
SOURCE=cifar100
TARGET=cifar10

# resnet / last layers

for MODEL in resnet56 densenet121; do
	for FEATURE_EXTRACTION in True False;do 
		for LR in 0.100 0.010 0.001; do
			cat  transfer.sh.template | sed -e "s/LR/$LR/;s/MODEL/$MODEL/;s/SOURCE/$SOURCE/;s/TARGET/$TARGET/;s/FEATURE_EXTRACTION/$FEATURE_EXTRACTION/" > "transfer/job${var}.sh"
			chmod +x "transfer/job${var}.sh"
			var=$((var+1))
		done
	done
done

cd transfer
python run_jobs_simple.py --num_parallel_jobs=12 --total_epochs=25