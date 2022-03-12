#!/bin/bash
hidden_dims=(32 64 128 256)
styles=("reformer" "original" "lsh")
for dims in "${hidden_dims[@]}"
do
    for style in "${styles[@]}"
    do
           echo "memorytest-$dims-$style"
           python train.py -n "memorytest-$dims-$style" --style=$style --hidden_size=$dims --num_epochs=1 & 
           wait
    done
done

python train.py -n "full-run-with-reformer" &
