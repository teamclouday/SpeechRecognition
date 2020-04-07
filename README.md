# SpeechRecognition

I'm trying to build an ASR model with tensorflow in this project  
May take a long time  

------

Dataset is from [openslr](http://www.openslr.org/12/)  
I'm using the 360 hours recordings  

------

I'm running the model on Google Cloud, with a Tesla T4 GPU  

------

### Note  
Performance per epoch is reduced by `librosa.load` greatly, need to pre-load audio files and dump the them into pickle before training  
