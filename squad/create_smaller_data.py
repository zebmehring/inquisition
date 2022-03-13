import numpy as np
import pdb

SIZE_OF_SMALLER_TRAIN_SET = 32#10000#32 #3 #100 #3 #1000 # this will be three batches

np.random.seed(224)

def shuffle_data(data):
    context_idxs=data['context_idxs'] 
    np.random.shuffle(context_idxs)
    
    context_char_idxs = data['context_char_idxs']
    np.random.shuffle(context_char_idxs)
    
    ques_idxs=data['ques_idxs']
    np.random.shuffle(ques_idxs)
    
    ques_char_idxs=data['ques_char_idxs']
    np.random.shuffle(ques_char_idxs)
    
    y1s=data['y1s']
    np.random.shuffle(y1s)
    
    y2s=data['y2s']
    np.random.shuffle(y2s)
    
    ids=data['ids']
    np.random.shuffle(ids)


    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs, y1s, y2s, ids




data = np.load('data/train.npz')


context_idxs, context_char_idxs, ques_idxs, ques_char_idxs, y1s, y2s, ids = shuffle_data(data)
np.savez('data/smaller_train.npz', context_idxs=context_idxs[:SIZE_OF_SMALLER_TRAIN_SET],
                                    context_char_idxs=context_char_idxs[:SIZE_OF_SMALLER_TRAIN_SET],
                                    ques_idxs=ques_idxs[:SIZE_OF_SMALLER_TRAIN_SET],
                                    ques_char_idxs=ques_char_idxs[:SIZE_OF_SMALLER_TRAIN_SET],
                                    y1s=y1s[:SIZE_OF_SMALLER_TRAIN_SET],
                                    y2s=y2s[:SIZE_OF_SMALLER_TRAIN_SET],
                                    ids=ids[:SIZE_OF_SMALLER_TRAIN_SET])

data = np.load('data/dev.npz')


context_idxs, context_char_idxs, ques_idxs, ques_char_idxs, y1s, y2s, ids = shuffle_data(data)
np.savez('data/smaller_dev.npz', context_idxs=context_idxs[:SIZE_OF_SMALLER_TRAIN_SET], 
                                    context_char_idxs=context_char_idxs[:SIZE_OF_SMALLER_TRAIN_SET],
                                    ques_idxs=ques_idxs[:SIZE_OF_SMALLER_TRAIN_SET],
                                    ques_char_idxs=ques_char_idxs[:SIZE_OF_SMALLER_TRAIN_SET],
                                    y1s=y1s[:SIZE_OF_SMALLER_TRAIN_SET],
                                    y2s=y2s[:SIZE_OF_SMALLER_TRAIN_SET],
                                    ids=ids[:SIZE_OF_SMALLER_TRAIN_SET])
