conda env update squad -f environment.yaml;
conda activate squad;
nohup python train.py -n get-results --num_epochs=1; # also add in the different training file here.
