?	˜.???%@˜.???%@!˜.???%@	u??AO??u??AO??!u??AO??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:˜.???%@7???????AKY?8?E%@Y34????rEagerKernelExecute 0*J7?A`?Z@)      =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?٭e2??!??5`B@)?\5???1{hۜ?@:Preprocessing2U
Iterator::Model::ParallelMapV2???????!z????0@)???????1z????0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate<?2T?T??!ط????7@)???????1???	q?+@:Preprocessing2F
Iterator::Model?????Ş?!??1??;@)?)??s??1?.??&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?£?#??!?$?$$@)?£?#??1?$?$$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?ȑ??ȳ?!
?3 R@)h????1㻌Kvk @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?óy?!?S?29?@)?J?óy?1?S?29?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapz?S?4???!,???6
;@)??	?yk?1???ǩ@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9u??AO??I??|a??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	7???????7???????!7???????      ??!       "      ??!       *      ??!       2	KY?8?E%@KY?8?E%@!KY?8?E%@:      ??!       B      ??!       J	34????34????!34????R      ??!       Z	34????34????!34????b      ??!       JCPU_ONLYYu??AO??b q??|a??X@Y      Y@q???r???"?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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