?	??]???)@??]???)@!??]???)@	?h?a????h?a???!?h?a???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??]???)@????6???A??&??)@YQN??????rEagerKernelExecute 0*	?n??2_@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???~?:??!???P?B@)?I??1nu?vb@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatey????ޡ?!W??L2?;@)7???????1g?8?1@:Preprocessing2U
Iterator::Model::ParallelMapV2??[??A??!F??ip+@)??[??A??1F??ip+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicex$(~???!?J]<?#@)x$(~???1?J]<?#@:Preprocessing2F
Iterator::Model]p????!?80')7@)=?U?????1?????O#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?p>????!?q?3?5S@)?j??Մ?1ת???M @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorT8?T?}?!????@)T8?T?}?1????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'?????!???J??>@)?ص?ݒl?1/???7\@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?h?a???I/?<???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????6???????6???!????6???      ??!       "      ??!       *      ??!       2	??&??)@??&??)@!??&??)@:      ??!       B      ??!       J	QN??????QN??????!QN??????R      ??!       Z	QN??????QN??????!QN??????b      ??!       JCPU_ONLYY?h?a???b q/?<???X@Y      Y@qq(|????"?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 