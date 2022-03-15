#!/bin/bash
styles=("reformer" "original" "lsh")

ques_limits=(64 128 192 256)
para_limits=(64 128 192 256)
for i in "${!ques_limits[@]}"
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
