# MxNet to TensorFlow converter

This is small project for converting some pretrained CNN models from MxNet format to TensorFlow. 
Some small changes are made based on vuvko's code

## Supported Layers

* **Activations**: ReLU
* **Batch normalization** without `use_global` flag (**Some changed are made**)
* **Convolution** without bias
* **Elementwise:** add
* **Flatten**
* **Fully connected** (**Some changed are made**)
* **Normalization:** l2
* **Pooling**: max, global pooling
* **Softmax** for output


## Supported Layers (Newly added)

* **LeakyReLU**: PReLU
* **mul_scalar** multipy
* **minus_scalar** minus
* **dropout:** dropout
