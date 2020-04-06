# this file will sort the training data by its length and output a sorted array of paths as a txt file  
import os
import pickle
import librosa

data = {}
TRAIN_DATA_PATH = os.path.join("LibriSpeech", "train-clean-360")
TEST_DATA_PATH = os.path.join("LibriSpeech", "test-clean")

counter = 0

def find_data(path=TRAIN_DATA_PATH, flag="Train"):
    global data
    global counter
    handle_data = False
    for item in os.listdir(path):
        new_path = os.path.join(path, item)
        if os.path.isdir(new_path):
            find_data(new_path, flag=flag)
        else:
            handle_data = True
            break
    if handle_data:
        children = os.listdir(path)
        txt_file = [x for x in children if x.endswith("txt")][0]
        with open(os.path.join(path, txt_file)) as inFile:
            for line in inFile.readlines():
                num, text = line.rstrip().split(" ", 1)
                filename = os.path.join(path, num+".flac")
                y, sr = librosa.load(filename, sr=16000)
                with open(os.path.join("dataset", flag, str(counter)+".pkl"), "wb") as outFile:
                    pickle.dump([y, sr], outFile)
                length = librosa.core.get_duration(y=y, sr=sr)
                filename = os.path.join("dataset", flag, str(counter)+".pkl")
                data[filename] = [length, text]
                print("{0} is loaded".format(filename))

def write_data(flag="train", sort=False):
    global data
    if sort:
        with open("{}_sorted.txt".format(flag), "w") as outFile:
            for key, val in sorted(data.items(), key=lambda x: x[1][0]):
                print("{0},{1},{2}".format(key, val[0], val[1]), file=outFile)
    else:
        with open("{}.txt".format(flag), "w") as outFile:
            for key, val in data.items():
                print("{0},{1},{2}".format(key, val[0], val[1]), file=outFile)

if __name__ == "__main__":
    if not os.path.exists(os.path.join("dataset", "train")):
        os.makedirs(os.path.join("dataset", "train"))
    if not os.path.exists(os.path.join("dataset", "test")):
        os.makedirs(os.path.join("dataset", "test"))
    find_data(path=TRAIN_DATA_PATH, flag="train")
    write_data(flag="train")
    data = {}
    counter = 0
    find_data(path=TEST_DATA_PATH, flag="test")
    write_data(flag="test")