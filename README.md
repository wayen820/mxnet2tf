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

## 解决mace 不支持matmul gpu操作方法
在tensorflow_converter.py 中替换  
TFOpType.MatMul.name: self.convert_matmul,  
TFOpType.BatchMatMul.name: self.convert_matmul,  
为  
TFOpType.MatMul.name: self.convert_fully_connected,  
TFOpType.BatchMatMul.name: self.convert_fully_connected,  
增加：  
def convert_fully_connected(self, tf_op):  
op = self.convert_general_op(tf_op)  
#param = tf_op.layer.inner_product_param  
op.type = MaceOp.FullyConnected.name  
