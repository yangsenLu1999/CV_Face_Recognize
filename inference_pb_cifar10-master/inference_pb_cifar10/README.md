# inference_pb_cifar10

对于我们之前的cifar10分类任务，如何导出pb文件，并利用pb文件进行inference推理。

在后面封装web服务的时候，我们也会介绍到。


重点：

# graph.py

output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                       sess.graph.as_graph_def(),
                                                       ['ArgMax'])

with tf.gfile.FastGFile('output_graph.pb', 'wb') as f:
    f.write(output_graph_def.SerializeToString())
    
通过上述代码保存pb文件。
其中，ArgMax是我们想要保存的graph，数据流最后的tensor的name，如果不清楚，则打印想要计算的tensor，就能够打印出他的name

此处的name，在后续inference.py文件中还会用到。

记住在保存pb文件的时候，一定要在restore之后


# inference.py

利用pb文件进行推理，其中output对应输出的tensor，input则对应到需要喂给网络的tensor。

   