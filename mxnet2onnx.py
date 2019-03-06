import cv2
import numpy as np
import sys
import argparse
import json
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import logging
logging.basicConfig(level=logging.INFO)




# Downloaded input symbol and params files
sym = './mobilefacenet-res4-8-16-4-dim512/model-symbol.json'
params = './mobilefacenet-res4-8-16-4-dim512/model-0000.params'
# sym = './resnet-18-symbol.json'
# params = './resnet-18-0000.params'


# Standard Imagenet input - 3 channels, 224*224
input_shape = (1,3,112,112)

# Path of the output file
onnx_file = './res512.onnx'

# Invoke export model API. It returns path of the converted onnx model
converted_model_path = onnx_mxnet.export_model(sym, params, [input_shape], np.uint8, onnx_file)

from onnx import checker
import onnx

# Load onnx model
model_proto = onnx.load_model(converted_model_path)

# Check if converted ONNX protobuf is valid
checker.check_graph(model_proto.graph)
