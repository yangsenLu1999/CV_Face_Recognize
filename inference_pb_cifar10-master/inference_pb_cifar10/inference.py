import tensorflow as tf
slim = tf.contrib.slim
import readcifar10
import os
import numpy as np

pb_path = "output_graph.pb"
output = "ArgMax:0"  
input_data = 'input_32:0'
##如果有多个input，需要feed多个值，在这里加tensor的name

img_data = np.zeros([1, 32, 32, 3])

with tf.Session() as sess:
    with tf.gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = sess.graph_def
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        pred = sess.graph.get_tensor_by_name(output)

        #次数为需要feed的值

        predictions = sess.run(pred,{input_data: img_data})
        
        print(predictions)