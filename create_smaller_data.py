import numpy as np

SIZE_OF_SMALLER_TRAIN_SET = 3 #100 #3 #1000




data = np.load('squad/data/train.npz')


np.savez('squad/data/smaller_train.npz', context_idxs=data['context_idxs'][:SIZE_OF_SMALLER_TRAIN_SET],
                                    context_char_idxs=data['context_char_idxs'][:SIZE_OF_SMALLER_TRAIN_SET],
                                    ques_idxs=data['ques_idxs'][:SIZE_OF_SMALLER_TRAIN_SET],
                                    ques_char_idxs=data['ques_char_idxs'][:SIZE_OF_SMALLER_TRAIN_SET],
                                    y1s=data['y1s'][:SIZE_OF_SMALLER_TRAIN_SET],
                                    y2s=data['y2s'][:SIZE_OF_SMALLER_TRAIN_SET],
                                    ids=data['ids'][:SIZE_OF_SMALLER_TRAIN_SET])

data = np.load('squad/data/dev.npz')


np.savez('squad/data/smaller_dev.npz', context_idxs=data['context_idxs'][:SIZE_OF_SMALLER_TRAIN_SET],
                                    context_char_idxs=data['context_char_idxs'][:SIZE_OF_SMALLER_TRAIN_SET],
                                    ques_idxs=data['ques_idxs'][:SIZE_OF_SMALLER_TRAIN_SET],
                                    ques_char_idxs=data['ques_char_idxs'][:SIZE_OF_SMALLER_TRAIN_SET],
                                    y1s=data['y1s'][:SIZE_OF_SMALLER_TRAIN_SET],
                                    y2s=data['y2s'][:SIZE_OF_SMALLER_TRAIN_SET],
                                    ids=data['ids'][:SIZE_OF_SMALLER_TRAIN_SET])
