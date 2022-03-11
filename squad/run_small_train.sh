# You can ru nthis script with an argument to save a special nmae 

python3 train.py -n train-small-${1:-default_name} --train_record_file 'data/smaller_train.npz' --eval_steps 30 --dev_record_file 'data/smaller_dev.npz'

#python test.py -n test-small-${1:-default_name} --dev_record_file 'data/smaller_dev.npz' --split "dev" --load_path "./save/train-small-${1:-default_name}"
