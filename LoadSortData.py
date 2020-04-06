# this file will sort the training data by its length and output a sorted array of paths as a txt file  
import os
import pickle
import librosa

data = {}
TRAIN_DATA_PATH = os.path.join("LibriSpeech", "train-clean-360")
TEST_DATA_PATH = os.path.join("LibriSpeech", "test-clean")

def find_data(path=TRAIN_DATA_PATH):
    global data
    handle_data = False
    for item in os.listdir(path):
        new_path = os.path.join(path, item)
        if os.path.isdir(new_path):
            find_data(new_path)
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
                length = librosa.core.get_duration(y=y, sr=sr)
                #filename = filename[3:]
                data[filename] = [length, text]
                print("{0} is loaded".format(filename))

def write_data(sort=False):
    global data
    if sort:
        with open("data_dict_sorted.txt", "w") as outFile:
            for key, val in sorted(data.items(), key=lambda x: x[1][0]):
                print("{0},{1},{2}".format(key, val[0], val[1]), file=outFile)
    else:
        with open("data_dict.txt", "w") as outFile:
            for key, val in data.items():
                print("{0},{1},{2}".format(key, val[0], val[1]), file=outFile)

if __name__ == "__main__":
    find_data(path=TEST_DATA_PATH)
    # with open("temp.pickle", "wb") as outFile:
    #     pickle.dump(data, outFile)
    write_data()
    # write_data(sort=True)