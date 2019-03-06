from tensorflow.python.platform import gfile
import tensorflow as tf 
import cv2
import numpy as np
import sys
import argparse
import json
import mxnet as mx

gpu_id = 0
# img = cv2.imread('test.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (112, 112,))
# img = np.swapaxes(img, 0, 2)
# img = np.swapaxes(img, 1, 2)
# image_npy = np.array([img])
image_npy=np.random.randint(0,256,1*3*112*112,np.uint8).reshape((1,3,112,112))
#image_npy=np.arange(1*3*12*12).reshape((1,3,12,12))
print(image_npy.shape)

tf_image_npy = np.transpose(image_npy,(0,2,3,1))

print(tf_image_npy.shape)


# load model
Session_config = tf.ConfigProto(allow_soft_placement = True)
Session_config.gpu_options.allow_growth=True 
sess = tf.Session(config=Session_config)
with gfile.FastGFile('tf-model-191.pb', 'rb') as f:
    with tf.device('/gpu:%d' %(gpu_id)):
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='') 

input_data = sess.graph.get_tensor_by_name('data:0')


js_model = json.load(open('./mobilefacenet-res4-8-16-4-dim512/model-symbol.json', 'r'))
mx_model, arg_params, aux_params = mx.model.load_checkpoint('./mobilefacenet-res4-8-16-4-dim512/model', 0)

nodes = js_model['nodes']

all_layers = mx_model.get_internals()
print(all_layers.list_outputs())

def show_difference(node):
    node_name = node['name']
    sym = all_layers[node_name + '_output']
    image_nd = mx.nd.array(image_npy * 1.0,ctx=mx.cpu())
    eval_iter = mx.io.NDArrayIter(data=image_nd,batch_size=1)
    mod = mx.mod.Module(symbol=sym,context=mx.cpu())
    mod.bind(for_training=False,data_shapes=eval_iter.provide_data)
    mod.set_params(arg_params,aux_params)
    batch_data = eval_iter.next()
    mod.forward(batch_data)
    mx_res = mod.get_outputs()
    mx_feat = mx_res[0].asnumpy()

    

    if(node_name[0] == '_' and node_name[1] == 'p'):
        tf_node_name = 'A' + node_name  + ':0'
    else:
        tf_node_name = node_name  + ':0'
    
    outputs = sess.graph.get_tensor_by_name(tf_node_name)
    tf_res = sess.run(outputs, feed_dict={input_data: tf_image_npy})
    #print('mxnet result: ', mx_feat.shape)
    #print(mx_feat)
    if(len(tf_res.shape) > 2):
        tf_res = np.transpose(tf_res,(0,3,1,2))
    #print('tensorflow result: ', tf_res.shape)
    #print(mx_feat)
    diff= abs(mx_feat - tf_res)
    #print(diff)
    print('Mean abs max error: ',np.max(diff))



for node_idx, node in enumerate(nodes):
    op = node['op']
    print('Parsing node %s with operator %s and index %d' % (node['name'], op, node_idx))
    if op == 'BatchNorm':
        show_difference(node)
    elif op == 'elemwise_add' or op == '_Plus':
        show_difference(node)
    elif op == 'Convolution':
        show_difference(node)
    elif op == 'LeakyReLU':
        show_difference(node)
    elif op == 'FullyConnected':
        show_difference(node)
    elif op == '_copy':
        show_difference(node)
    elif op=='SoftmaxActivation':
        show_difference(node)


