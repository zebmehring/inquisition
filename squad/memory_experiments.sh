#!/bin/bash
hidden_dims=(32 64 128 256)
styles=("reformer" "original" "lsh")
batch_sizes=(4 8 16)
for dims in "${hidden_dims[@]}"
do
    for style in "${styles[@]}"
    do
       for batch_size in "${batch_sizes[@]}"
       do
           echo "memorytest-$dims-$style-$batch_size"
           python train.py -n "memorytest-$dims-$style-$batch_size" --style=$style --hidden_size=$dims --batch_size=$batch_size --num_epochs=1 & 
           wait
       done       
    done
done

python train.py -n "full-run-with-reformer" &
