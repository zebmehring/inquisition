#!/bin/bash
hidden_dims=(32 64 128 256 512)
styles=("reformer" "original" "lsh")
for dims in "${hidden_dims[@]}"
do
    for style in "${styles[@]}"
    do
           echo "memorytest-$dims-$style"
           python train.py -n "memorytest-$dims-$style" --style=$style --hidden_size=$dims --num_epochs=1 --train_record_file 'data/smaller_train.npz' --dev_record_file 'data/smaller_dev.npz' & 
           wait
    done
done

ques_limits=(20 30 40 50)
para_limits=(100 200 300 400)
for i in "${!foo[@]}"
do 
    ques_limit=${ques_limits[$i]}
    para_limit=${para_limits[$i]}
    python setup.py --ques_limit=$ques_limit --para_limit=$para_limit &
    wait
    python create_smaller_data.py &
    wait 
    for style in "${styles[@]}"
    do
           echo "memorytest-$style-$ques_limit-$para_limit"
           python train.py -n "memorytest-$style-$ques_limit-$para_limit" --style=$style --num_epochs=1 --train_record_file 'data/smaller_train.npz' --dev_record_file 'data/smaller_dev.npz' --eval_steps=16 &
           wait
    done
done
