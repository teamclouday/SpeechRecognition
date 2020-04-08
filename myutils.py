# This file contain all the classes and functions for preprocessing the data
# code adopted from https://keras.io/examples/image_ocr/

import os
import math
import time
import pickle
import librosa
import itertools
import editdistance
import numpy as np
import pandas as pd
import tensorflow as tf
from jiwer import wer

FREQ = 16000

# Text data preprocessing functions
CHAR_LIST = "".join([" "] + [chr(x) for x in range(65, 91)] + ["'"])

def text_to_label(text):
    """Convert text to sequence"""
    text = list(text)
    result = []
    for ch in text:
        result.append(CHAR_LIST.find(ch))
    return result

def label_to_text(label):
    """Convert sequence to text"""
    result = []
    for x in label:
        if x == len(CHAR_LIST):
            result.append("")
        else:
            result.append(CHAR_LIST[x])
    return "".join(result)

# a generator for training data
class AudioTextGenerator(tf.keras.callbacks.Callback):
    def __init__(self, data_dict_path,
                 batch_size, max_audio_length,
                 feature_height,
                 # val_split,
                 shuffle_random_state=None,
                 max_text_length=None,
                 feature_choice="mfcc"):
        data = pd.read_csv(data_dict_path)
        data = data[data['Length'] < max_audio_length] # remove audio data that are longer
        self.data = data.sample(frac=1, random_state=shuffle_random_state).reset_index(drop=True)
        print("AudioTextGenerator: self.data.shape = {}".format(self.data.shape))
        self.shuffle_random_state = shuffle_random_state
        self.batch_size = batch_size
        self.max_audio_length = math.ceil(max_audio_length * FREQ / 512)
        self.feature_height = feature_height
        self.feature_choice = feature_choice
        #assert val_split < 0.5
        #assert val_split > 0
        #self.val_split = val_split
        self.train_index_pos = 0
        #self.val_index_pos = math.floor(len(self.data)*(1-self.val_split))
        self.train_end_pos = len(self.data) - 1 #self.val_index_pos - 1
        if max_text_length is None:
            self.max_text_length = self.data["Text"].str.len().max()
        else:
            self.max_text_length = max_text_length

    # generate text labels for each batch
    # returns padded data, and actual sequence length
    def prepare_labels(self, index_list):
        assert hasattr(self, "data")
        Y_text_source = self.data["Text"].iloc[index_list].copy().to_list()
        Y_text = [text_to_label(x) for x in Y_text_source]
        sequence_length = [len(x) for x in Y_text]
        Y_data = tf.keras.preprocessing.sequence.pad_sequences(Y_text, padding="post", maxlen=self.max_text_length)
        return (Y_data, sequence_length, Y_text_source)
    
    # generate audio data for each batch
    # returns padded data, and actual sequence length
    def prepare_audio(self, index_list):
        assert hasattr(self, "data")
        X_path = self.data["Path"].iloc[index_list].copy().to_list()
        X_data = []
        for path in X_path:
            with open(path, "rb") as inFile:
                X_data.append(pickle.load(inFile))
        if self.feature_choice == "mfcc":
            X_data = [librosa.feature.mfcc(y=a, sr=b, n_mfcc=self.feature_height).T for (a, b) in X_data]
            sequence_length = [x.shape[0] for x in X_data]
        else:
            X_data = [librosa.feature.melspectrogram(y=a, sr=b, n_mels=self.feature_height).T for (a, b) in X_data]
            sequence_length = [x.shape[0] for x in X_data]
        return (tf.keras.preprocessing.sequence.pad_sequences(X_data, maxlen=self.max_audio_length, padding="post"), sequence_length)
    
    # helper function to get a batch
    # either for training or validation
    def get_batch(self, index, size, train):
        #start_time = time.time()
        if (index + size) >= len(self.data):
            size = len(self.data) - 1 - index
        if train and (index + size) > self.train_end_pos:
            size = self.train_end_pos - index
        index_list = np.arange(index, index+size)
        X_data, input_length = self.prepare_audio(index_list)
        Y_data, label_length, source_str = self.prepare_labels(index_list)
        inputs = {
            'the_input': X_data[:,:,:,np.newaxis],
            'the_labels': np.array(Y_data, dtype=np.float32),
            'input_length': np.array(input_length)[:,np.newaxis],
            'label_length': np.array(label_length)[:,np.newaxis],
            'source_str': np.array(source_str)
        }
        outputs = {'ctc': np.zeros([size])}
        #print("{:.2f}s".format(time.time() - start_time))
        return (inputs, outputs)

    # function called in training
    def next_batch(self):
        while True:
            result = self.get_batch(self.train_index_pos, self.batch_size, train=True)
            self.train_index_pos += self.batch_size
            if self.train_index_pos > self.train_end_pos:
                self.train_index_pos = 0 # if exceeds, reset the position
                # self.val_index_pos = self.train_end_pos + 1
                self.data = self.data.sample(frac=1, random_state=self.shuffle_random_state).reset_index(drop=True) # and shuffle data
            yield result

    # function called in validation
    #def next_val(self):
    #    while True:
    #        result = self.get_batch(self.val_index_pos, self.batch_size, train=False)
    #        self.val_index_pos += self.batch_size
    #        if self.val_index_pos >= len(self.data):
    #            self.val_index_pos = self.train_end_pos + 1 # if exceeds, just reset itself
    #        yield result

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        self.data = self.data.sample(frac=1, random_state=self.shuffle_random_state).reset_index(drop=True) # shuffle data

# ctc function
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = label_to_text(out_best)
        ret.append(outstr)
    return ret

# define callback function on validation
class ValCallback(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_path, test_func, next_val, num_display):
        self.test_func = test_func
        self.ckpt_path = ckpt_path
        self.next_val = next_val
        self.num_display = num_display
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
    
    #def show_edit_distance(self, num):
    #    num_left = num
    #    mean_norm_ed = 0.0
    #    mean_ed = 0.0
    #    while num_left > 0:
    #        word_batch = next(self.next_val)
    #        num_proc = min(word_batch['the_input'].shape[0], num_left)
    #        decoded_res = decode_batch(self.test_func,
    #                                   word_batch['the_input'][0:num_proc])
    #        for j in range(num_proc):
    #            edit_dist = editdistance.eval(decoded_res[j],
    #                                          word_batch['source_str'][j])
    #            mean_ed += float(edit_dist)
    #            mean_norm_ed += float(edit_dist) / len(word_batch['source_str'][j])
    #        num_left -= num_proc
    #    mean_norm_ed = mean_norm_ed / num
    #    mean_ed = mean_ed / num
    #    print('\nOut of %d samples:  Mean edit distance:'
    #          '%.3f Mean normalized edit distance: %0.3f'
    #          % (num, mean_ed, mean_norm_ed))

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(
            os.path.join(self.ckpt_path, 'weights%02d.h5' % (epoch)))
        #self.show_edit_distance(50)
        word_batch = next(self.next_val)[0]
        res = decode_batch(self.test_func, word_batch['the_input'])
        print()
        for i in range(self.num_display):
            print("Truth = {}\nDecoded = {}".format(word_batch["source_str"][i], res[i]))
        print("WER: {}".format(wer(word_batch["source_str"].tolist(), res)))