from tensorflow.python.platform import gfile
import tensorflow as tf 
import cv2
import numpy as np
import sys
import argparse
import json
import mxnet as mx

gpu_id = 0
img = cv2.imread('test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (112, 112,))
img = np.swapaxes(img, 0, 2)
img = np.swapaxes(img, 1, 2)
image_npy = np.array([img])
print(image_npy.shape)

tf_image_npy = np.transpose(image_npy,(0,2,3,1))


t=tf_image_npy.astype(np.float32)
print(t.dtype)
t.tofile("./d.bin")
print(t.reshape(-1)[:100])

# load model
Session_config = tf.ConfigProto(allow_soft_placement = True)
Session_config.gpu_options.allow_growth=True 
sess = tf.Session(config=Session_config)
with gfile.FastGFile('tf_model_opt.pb', 'rb') as f:
    with tf.device('/gpu:%d' %(gpu_id)):
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='') 

input_data = sess.graph.get_tensor_by_name('data:0')
outputs = sess.graph.get_tensor_by_name('pre_fc1:0')
tf_res = sess.run(outputs, feed_dict={input_data: tf_image_npy})

print(tf_res)
a=np.array([-1.23242,2.75977,7.27344,2.21289,3.2793,-2.83789,1.125,9.60156,3.44727,-3.99219,9.72656,-2.14844,-2.5293,4.92969,4.97656,-3.08984,-2.90234,-1.59375,-5.43359,-0.170776,1.52832,-3.75977,3.80469,1.13574,6.35156,2.58789,-1.73438,-1.17871,3.38281,5.44531,3.65625,8.83594,-5.69922,6.78125,2.53711,-3.6875,0.0852051,-1.72168,-4.55859,0.937988,-5.44531,4.26562,2.59766,1.5127,-3.12891,2.66016,-2.91406,-6.52344,1.05566,1.06348,-0.544434,1.90137,-3.14453,3.61914,1.4248,-3.23047,-2.00781,-4.89844,6.21094,-2.06445,0.483154,4.78125,1.44336,-5.00781,2.72461,-2.99609,6.75391,-0.606445,4.94531,-4.61328,-3.22656,-3.69531,-1.84473,8.53906,5.26172,-4.1875,4.49609,-2.36719,-1.4502,-3.98633,-0.606445,-5.94531,2.65234,8.90625,2.57812,-2.93945,7.82812,-6.91406,0.869141,-4.92969,1.17871,-7.95312,-4.20703,2.23438,-6.33594,-1.1875,-1.26562,4.76562,-1.4668,-0.335449,-2.9375,2.51562,-2.12695,3.4043,3.95117,-4.08984,2.00195,-10.6406,2.21289,8.16406,-4.27344,9.21875,0.695801,-5.94141,-5.31641,-6.99609,-1.99023,-7.33984,3.49609,-3.87695,-8.91406,0.638184,9.36719,12.9688,-2.28516,-4.93359,1.3252,3.5957])
vector1=tf_res.reshape(-1)
vector2=a
op7=np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
print(op7)

