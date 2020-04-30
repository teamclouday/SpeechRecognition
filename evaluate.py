import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import wave
import librosa
import pyaudio
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from myutils import FREQ, HOP, WINDOW, CHAR_LIST, label_to_text

max_audio_duration = 12 # in seconds
feature_height = 40

conv_filters = 16
kernel_size = (3, 3)
pool_size = 2
time_dense_size = 80
rnn_size = 512
max_audio_length = 1205

tf.keras.backend.clear_session()
input_data = tf.keras.layers.Input(name="the_input", shape=(max_audio_length, feature_height), dtype="float32")
inner = tf.keras.layers.Reshape(target_shape=(max_audio_length, feature_height, 1), name="reshape1")(input_data)
inner = tf.keras.layers.Conv2D(conv_filters, kernel_size, padding="same", activation="relu", kernel_initializer="he_normal", name="conv1")(inner)
inner = tf.keras.layers.MaxPool2D(pool_size=(1, pool_size), name="max1")(inner)
inner = tf.keras.layers.Conv2D(conv_filters, kernel_size, padding="same", activation="relu", kernel_initializer="he_normal", name="conv2")(inner)
inner = tf.keras.layers.MaxPool2D(pool_size=(1, pool_size), name="max2")(inner)
inner = tf.keras.layers.Reshape(target_shape=(max_audio_length, (feature_height // (pool_size ** 2)) * conv_filters), name="reshape2")(inner)
inner = tf.keras.layers.Dense(time_dense_size, activation="relu", name="dense1")(inner)
gru_1 = tf.keras.layers.GRU(rnn_size, return_sequences=True, kernel_initializer="he_normal", name="gru1")(inner)
gru_1b = tf.keras.layers.GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer="he_normal", name="gru1_b")(inner)
gru1_merged = tf.keras.layers.add([gru_1, gru_1b])
gru_2 = tf.keras.layers.GRU(rnn_size, return_sequences=True, kernel_initializer="he_normal", name="gru2")(gru1_merged)
gru_2b = tf.keras.layers.GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer="he_normal", name="gru2_b")(gru1_merged)
inner = tf.keras.layers.Dense(len(CHAR_LIST) + 1, kernel_initializer="he_normal", name="dense2")(tf.keras.layers.concatenate([gru_2, gru_2b]))
y_pred = tf.keras.layers.Activation('softmax', name='softmax')(inner)

model = tf.keras.Model(inputs=input_data, outputs=y_pred)

print(model.summary())

model.load_weights(os.path.join("weights.h5"))

print("Model loaded")

run = True
while run:
    if(input("Press Enter to record (q to exit): ") == "q"):
        break
    audio = pyaudio.PyAudio()
    stream = audio.open(format=audio.get_format_from_width(2), channels=1, rate=FREQ, input=True, output=True, frames_per_buffer=HOP)
    new_audio = []
    print("Recording Started ({} sec)".format(max_audio_duration))
    for _ in range(int(FREQ / HOP * max_audio_duration)):
        new_audio.append(stream.read(HOP))
    print("Recording Stopped")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    with wave.open("recording.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(FREQ)
        wf.writeframes(b''.join(new_audio))
        print("Saved to file")
    audio, sr = librosa.load("recording.wav", sr=FREQ)
    audio = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=feature_height, hop_length=HOP, n_fft=WINDOW).T
    audio_length = audio.shape[0]
    audio = tf.keras.preprocessing.sequence.pad_sequences([audio], maxlen=max_audio_length, padding="post", truncating="post")[0]
    audio = audio.reshape(1, -1, feature_height)
    prediction = model.predict(x=audio)
    result = tf.keras.backend.ctc_decode(prediction, np.array([audio_length], dtype=np.int32), greedy=True, beam_width=100, top_paths=1)[0][0]
    result = tf.keras.backend.get_value(result)[0]
    print("Raw Result:\n{}".format(result))
    result = label_to_text(result)
    print("Converted Result:\n{}".format(result))
    print()