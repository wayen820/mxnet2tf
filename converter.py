import tensorflow as tf
import numpy as np


class Converter(object):

    def __init__(self, tf_nodes, mx_nodes, mx_params):
        self.tf_nodes = tf_nodes
        self.mx_nodes = mx_nodes
        self.mx_params = mx_params

    def to_tuple(self, string, conv_type=str):
        return tuple(map(conv_type, map(str.strip, string[1:-1].split(','))))

    def create_var(self, node, shape=None):
        node_name = node['name']
        if shape is None:
            if node_name in self.mx_params:
                shape = self.mx_params[node_name].shape
            else:
                shape = ()
        # print('Creating var with shape:', shape)
        created_node = tf.get_variable(node_name, shape=shape, initializer=tf.zeros_initializer)
        self.tf_nodes[node_name] = created_node
        # if node_name in params:
        #     tf_nodes[node_name].load(params[node_name].asnumpy())
        return created_node

    def create_bn(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        if node['attrs'].get('eps') is not None:
            epsilon = float(node['attrs']['eps'])
        else:
            epsilon=0.001
        input_shape = input_sym.get_shape()
        axis = list(range(len(input_shape) - 1))

        def create_bn_params(i):
            cur_node = self.mx_nodes[node['inputs'][i][0]]
            cur_name = cur_node['name']
            self.create_var(cur_node)
            self.tf_nodes[cur_name].load(self.mx_params[cur_name].asnumpy())
            return self.tf_nodes[cur_name]
        if len(node['inputs']) > 3:
            gamma, beta, mean, var = (create_bn_params(i) for i in range(1, 5))
        else:
            gamma, beta = (create_bn_params(i) for i in range(1, 3))
            mean = tf.get_variable(node_name + '_mean', shape=input_shape[-1], initializer=tf.zeros_initializer)
            mean.load(np.zeros((input_shape[-1],), dtype='float32'))
            var = tf.get_variable(node_name + '_var', shape=input_shape[-1], initializer=tf.ones_initializer)
            var.load(np.ones((input_shape[-1],), dtype='float32'))
        # TODO: add support for swtiching between train and inference phases
        # For inference use_global_stats=False is ignored
        #
        # if 'use_global_stats' in node['attrs']:
        #     if node['attrs']['use_global_stats'] == 'False':
        #         # print('Not use')
        #         mean, var = tf.nn.moments(input_sym, axis)
        # else:
        #     mean, var = tf.nn.moments(input_sym, axis)
        if 'fix_gamma' in node['attrs']:
            if node['attrs']['fix_gamma'] == 'True':
                # print('Fix')
                gamma = tf.get_variable(node_name + '_gamma_fixed', shape=input_shape[-1], initializer=tf.ones_initializer)
                gamma.load(np.ones((input_shape[-1],), dtype='float32'))
        else:
            gamma = tf.get_variable(node_name + '_gamma_fixed', shape=input_shape[-1], initializer=tf.ones_initializer)
            gamma.load(np.ones((input_shape[-1],), dtype='float32'))
        if len(input_shape)==2:
            if input_sym.op.type=='Squeeze':
                print('发现二维输入的BN层，检查到上一层为squeeze，尝试将其往后移动并插入bn')
                bn_tensor = tf.nn.batch_normalization(input_sym.op.inputs[0], mean, var, beta, gamma, epsilon, name='bn' + node_name)
                bn_tensor=tf.squeeze(bn_tensor,[1,2],name='squeeze'+node_name)
            else:
                print('对不起，mace 目前不支持二维的BN')
                exit(0)
        else:
            bn_tensor = tf.nn.batch_normalization(input_sym, mean, var, beta, gamma, epsilon, name='bn' + node_name)
        self.tf_nodes[node_name] = tf.identity(bn_tensor,name = node_name)
        return self.tf_nodes[node_name]

    def create_conv_test(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        num_filters_in = input_sym.get_shape()[-1]
        num_filters_out = int(node['attrs']['num_filter'])
        kernel_size = self.to_tuple(node['attrs']['kernel'], int)
        # TODO: add bias support
        add_bias = node['attrs']['no_bias'] != 'True'
        if 'num_group' in node['attrs']:
            num_group = int(node['attrs']['num_group'])
        else:
            num_group = 1
        if 'pad' in node['attrs']:
            padding = self.to_tuple(node['attrs']['pad'], int)
        else:
            padding = (0, 0)
        stride = self.to_tuple(node['attrs']['stride'], int)
        if padding[0] > 0 or padding[1] > 0:
            padded_input = tf.pad(input_sym, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]],
                                  'CONSTANT')
        else:
            padded_input = input_sym
        #convolve = lambda input_sym, kernel, name=None: tf.nn.conv2d(input_sym, kernel, [1, stride[0], stride[1], 1],
        #                                                padding='VALID', name=name)
        if add_bias:
            bias_node=self.mx_nodes[node['inputs'][2][0]]
            #weights_node = self.mx_nodes[node['inputs'][2][0]]
            bias=self.create_var(bias_node,shape=(num_filters_out,))
            bias_numpy=self.mx_params[bias_node['name']].asnumpy()
            bias.load(bias_numpy)
        weights_node = self.mx_nodes[node['inputs'][1][0]]
        if num_group==1:
            weights = self.create_var(weights_node,
                                      shape=(kernel_size[0], kernel_size[1], num_filters_in // num_group,num_filters_out))
            weights_numpy = self.mx_params[weights_node['name']].asnumpy().transpose((2, 3, 1, 0))

        else:
            weights = self.create_var(weights_node,
                                      shape=(kernel_size[0], kernel_size[1],num_filters_out, num_filters_in // num_group))
            weights_numpy = self.mx_params[weights_node['name']].asnumpy().transpose((2, 3, 0, 1))
        weights.load(weights_numpy)
        if num_group > 1:
            if weights_numpy.shape[3] != 1:
                print('error:mace only support multiplier = 1')
                exit(0)
            else:
                self.tf_nodes[node_name] = tf.nn.depthwise_conv2d(input=padded_input, filter=weights,strides=[1, stride[0], stride[1], 1],padding='VALID',rate=[1,1],name=node_name)
        else:
            if add_bias:
                temp=tf.nn.conv2d(padded_input, weights, [1, stride[0], stride[1], 1],
                         padding='VALID')
                self.tf_nodes[node_name]=tf.nn.bias_add(temp,bias, name=node_name)
            else:
                self.tf_nodes[node_name] = tf.nn.conv2d(padded_input, weights, [1, stride[0], stride[1], 1],
                                                        padding='VALID',name=node_name)
            #self.tf_nodes[node_name] = convolve(padded_input, weights, name=node_name)
        return self.tf_nodes[node_name]
    def create_conv(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        num_filters_in = input_sym.get_shape()[-1]
        num_filters_out = int(node['attrs']['num_filter'])
        kernel_size = self.to_tuple(node['attrs']['kernel'], int)
        # TODO: add bias support
        # add_bias = node['attrs']['no_bias'] != 'True'
        if 'num_group' in node['attrs']:
            num_group = int(node['attrs']['num_group'])
        else:
            num_group = 1
        if 'pad' in node['attrs']:
            padding = self.to_tuple(node['attrs']['pad'], int)
        else:
            padding = (0, 0)
        stride = self.to_tuple(node['attrs']['stride'], int)
        weights_node = self.mx_nodes[node['inputs'][1][0]]
        weights = self.create_var(weights_node,
                             shape=(kernel_size[0], kernel_size[1], num_filters_in // num_group, num_filters_out))
        weights_numpy = self.mx_params[weights_node['name']].asnumpy().transpose((2, 3, 1, 0))
        if padding[0] > 0 or padding[1] > 0:
            padded_input = tf.pad(input_sym, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]], 'CONSTANT')
        else:
            padded_input = input_sym
        convolve = lambda input_sym, kernel, name=None: tf.nn.conv2d(input_sym, kernel, [1, stride[0], stride[1], 1], padding='VALID', name=name)
        weights.load(weights_numpy)
        if num_group > 1:
            input_groups = tf.split(axis=3, num_or_size_splits=num_group, value=padded_input)
            weight_groups = tf.split(axis=3, num_or_size_splits=num_group, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
            self.tf_nodes[node_name] = tf.concat(axis=3, values=output_groups, name=node_name)
        else:
            self.tf_nodes[node_name] = convolve(padded_input, weights, name=node_name)
        return self.tf_nodes[node_name]

    def create_pooling(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        pooling_type = node['attrs']['pool_type']
        kernel_size = self.to_tuple(node['attrs']['kernel'], int)
        pooling_convertion=node['attrs']['pooling_convention']
        if pooling_convertion=='FULL':
            print('create_pooling:sorry tensorflow do not support full padding')
            exit(0)
        if 'stride' in node['attrs']:
            stride = self.to_tuple(node['attrs']['stride'], int)
        else:
            stride = (1, 1)
        if 'global_pool' in node['attrs']:
            global_pool = node['attrs']['global_pool'] == 'True'
        else:
            global_pool = False
        if 'pad' in node['attrs']:
            padding = self.to_tuple(node['attrs']['pad'], int)
        else:
            padding = (0, 0)
        if global_pool:
            self.tf_nodes[node_name] = tf.reduce_mean(input_sym, reduction_indices=[1, 2], name=node_name)
        else:
            if padding[0] > 0 or padding[1] > 0:
                padded_input = tf.pad(input_sym,
                                      [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]],
                                      'CONSTANT')
            else:
                padded_input = input_sym
            if pooling_type == 'max':
                self.tf_nodes[node_name] = tf.nn.max_pool(padded_input,
                                                          ksize=[1, kernel_size[0], kernel_size[1], 1],
                                                          strides=[1, stride[0], stride[1], 1],
                                                          padding='VALID' if pooling_convertion=='valid' else 'SAME', name=node_name)
            else:
                raise NameError('Unknown pooling type: %s' % pooling_type)
        return self.tf_nodes[node_name]

    def create_activation(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        activation_type = node['attrs']['act_type']
        # TODO: more activation types
        if activation_type == 'relu':
            activation_fn = tf.nn.relu
        else:
            raise NameError('Unknown activation type: %s' % activation_type)
        self.tf_nodes[node_name] = activation_fn(input_sym, name=node_name)
        return self.tf_nodes[node_name]

    def create_softmax(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        self.tf_nodes[node_name] = tf.nn.softmax(input_sym, name=node_name)
        return self.tf_nodes[node_name]

    def create_softmaxActivation(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        mode = node['attrs']['mode']
        if mode=='instance':
            self.tf_nodes[node_name] = tf.nn.softmax(input_sym, name=node_name,axis=0)
        elif mode=='channel':
            self.tf_nodes[node_name] = tf.nn.softmax(input_sym, name=node_name)
        return self.tf_nodes[node_name]

    def create_elementwise(self, node, op='sum'):
        node_name = node['name']
        inputs_sym = [self.tf_nodes[self.mx_nodes[n[0]]['name']] for n in node['inputs']]
        if len(inputs_sym)==2:
            self.tf_nodes[node_name]=tf.add(inputs_sym[0],inputs_sym[1],name='A'+node_name)
        else:
        # TODO: more elementwise types
            if op == 'sum':
                self.tf_nodes[node_name] = tf.add_n(inputs_sym, name='A' + node_name)
            else:
                raise NameError('Unknown elementwise type: %s' % op)
        return self.tf_nodes[node_name]
    def create_fc_test(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        real_shape = input_sym.get_shape()
        if(len(real_shape) > 2):
            print('flatten in fc layer')
            # input_sym = tf.transpose(input_sym,(0,3,1,2))
            # input_sym=input_sym[..., 0, 0]
            # input_sym=tf.squeeze(input_sym,[1,2])
            # input_sym=tf.strided_slice(input_sym,)
            # input_sym = tf.contrib.layers.flatten(input_sym)
        num_units_in = input_sym.get_shape()[3]
        num_units_out = int(node['attrs']['num_hidden'])
        weights_node = self.mx_nodes[node['inputs'][1][0]]
        weights = self.create_var(weights_node, shape=(1,1,num_units_in, num_units_out))
        bias_node = self.mx_nodes[node['inputs'][2][0]]
        bias = self.create_var(bias_node, shape=(num_units_out,))
        weights_numpy = self.mx_params[weights_node['name']].asnumpy()
        weights.load(np.expand_dims(np.expand_dims(weights_numpy.T,axis=0),axis=0))
        bias.load(self.mx_params[bias_node['name']].asnumpy())
        # self.tf_nodes[node_name]=tf.matmul(input_sym,weights)+bias
        self.tf_nodes[node_name] = tf.squeeze(tf.nn.xw_plus_b(input_sym, weights, bias,name=node_name), [1,2],name='squeeze_'+node_name)
        return self.tf_nodes[node_name]

    def create_fc(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        real_shape = input_sym.get_shape()
        if(len(real_shape) > 2):
            print('flatten in fc layer')
            # input_sym = tf.transpose(input_sym,(0,3,1,2))
            input_sym = tf.contrib.layers.flatten(input_sym)
        num_units_in = input_sym.get_shape()[1]
        num_units_out = int(node['attrs']['num_hidden'])
        weights_node = self.mx_nodes[node['inputs'][1][0]]
        weights = self.create_var(weights_node, shape=(num_units_in, num_units_out))
        bias_node = self.mx_nodes[node['inputs'][2][0]]
        bias = self.create_var(bias_node, shape=(num_units_out,))
        weights_numpy = self.mx_params[weights_node['name']].asnumpy()
        weights.load(weights_numpy.T)
        bias.load(self.mx_params[bias_node['name']].asnumpy())
        self.tf_nodes[node_name] = tf.nn.xw_plus_b(input_sym, weights, bias, name=node_name)

        return self.tf_nodes[node_name]
    def void_test(self):
        pass

    def create_norm(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        self.tf_nodes[node_name] = tf.nn.l2_normalize(input_sym, dim=1, name=node_name)
        return self.tf_nodes[node_name]

    def create_flatten(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        self.tf_nodes[node_name] = tf.contrib.layers.flatten(input_sym)
        return self.tf_nodes[node_name]

    def create_dropout(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        keep_prob = 1.0 
        self.tf_nodes[node_name] = tf.nn.dropout(input_sym,keep_prob=keep_prob, name=node_name)
        return self.tf_nodes[node_name]

    def create_minus_scalar(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        scalar = float(node['attrs']['scalar'])
        #scalar_tensor = tf.fill(input_sym.get_shape(),scalar)
        self.tf_nodes[node_name] = input_sym - scalar #tf.subtract(input_sym,scalar_tensor,name = node_name)
        return self.tf_nodes[node_name]

    def create_mul_scalar(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        scalar = float(node['attrs']['scalar'])
        #scalar_tensor = tf.fill(input_sym.get_shape(),scalar)
        self.tf_nodes[node_name] = input_sym * scalar #tf.multiply(input_sym,scalar_tensor,name = node_name)
        return self.tf_nodes[node_name]

    def create_copy(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        self.tf_nodes[node_name] = tf.identity(input_sym,name = node_name)
        return self.tf_nodes[node_name]


    def create_LeakyReLU(self, node):
        node_name = node['name']
        input_sym = self.tf_nodes[self.mx_nodes[node['inputs'][0][0]]['name']]
        
        alpha_node = self.mx_nodes[node['inputs'][1][0]]
        alpha_name = alpha_node['name']
        self.create_var(alpha_node)
        self.tf_nodes[alpha_name].load(self.mx_params[alpha_name].asnumpy())
        alpha_para = self.tf_nodes[alpha_name]
        
        pos = tf.nn.relu(input_sym)
        neg = alpha_para * (input_sym - abs(input_sym)) * 0.5
        self.tf_nodes[node_name] = tf.add(pos,neg,name = node_name)
        return self.tf_nodes[node_name]





        


