# Speech Recognition  

I'm trying to build an ASR model with tensorflow 2.1 in this project  
May take a long time  

------

### Structure  

CNN -> Hidden Dense -> GRU(Add & Concat) -> Dense(softmax)  
Similar structure as [keras OCR example](https://keras.io/examples/image_ocr/)  
Optimizer using Adam, with lr = 0.001  
Batch size = 128  

------

### Data  

Dataset is from [openslr](http://www.openslr.org/12/)  
I'm using the 360 hours recordings  

------

### Training Env  

I'm currently training the model on Google Cloud, with a Tesla T4 GPU  

------

### Notes  
* Performance per epoch is reduced by `librosa.load` greatly, need to pre-load audio files and dump the them into pickle before training  
* The training data could be clean (clean voice), and data augmentation (noise, speed change, pitch change) need to be implemented  
* The input time steps length for RNN need to be greater than the length of converted label length  
* If loss becomes ```inf```, it means input length is not long enough to map to the target labels, and ctc loss output 0 probabilities  
* Can also rewrite the ```tf.keras.backend.ctc_batch_cost``` function to set ```ignore_longer_outputs_than_inputs```=```True```  
