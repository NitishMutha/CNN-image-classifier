
### P1: MNIST with TensorFlow  
#### a) 1 linear layer, followed by a softmax
code: tf_P1_a.py  

#### b) 1 hidden layer (128 units) with a ReLU non-linearity, followed by a softmax
code: tf_P1_b.py  

#### c) 2 hidden layers (256 units) each, with ReLU non-linearity, follow by a softmax
code: tf_P1_c.py  

#### d) 3 layer convolutional model (2 convolutional layers followed by max pooling) + 1 non-linear layer (256 units), followed by softmax.  
code: tf_P1_c.py  


### P2: MNIST without TensorFlow  
#### a) Derivations in report
#### b) Implement and train the model in (P1:a)
code: tf_P2_a.py  

#### c) Implement and train the model in (P1:b)
code: tf_P2_b.py  

#### d) Implement and train the model in (P1:c)
code: tf_P2_c.py  

**All the final trained weights can be found in the trained_weights folder**

### Setup to run source code  
1. Install TensorFlow on Anaconda environment (gpu version prefered for speed of execution), [setup for windows](https://nitishmutha.github.io/tensorflow/2017/01/22/TensorFlow-with-gpu-for-windows.html)
2. Install numpy, sklearn, matplotlib if not installed by default.  
3. Activate tensforflow environment. e.g. `activate tensorflow-gpu`
3. Navigate to source code directory and run each python file. (pycharm prefered)


**Note: Each python file has a flag called TRAIN_MODE. By default it is set to False to run code in TEST mode with saved parameters. You can toggle it to run in TRAIN mode**  
P.S. The saved parameters have been trained using tensorflow-gpu version r0.12.

 
