?	,?S?X#@,?S?X#@!,?S?X#@	'I5(x??'I5(x??!'I5(x??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:,?S?X#@?^?????A7o??#@Y??q?j???rEagerKernelExecute 0*	????M"U@2U
Iterator::Model::ParallelMapV2Oʤ?6 ??!?f4?1?@)Oʤ?6 ??1?f4?1?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??9?٘?!?:	?^?<@)?^?iN^??1???5??7@:Preprocessing2F
Iterator::ModelEKO???!?op??,G@)t?//?>??1\?X??Q.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate,f??!??!aMJSޡ2@)?j?????1e??!g)'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?????jx?!?b?	?4@)?????jx?1?b?	?4@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensort&m??q?!K???&?@)t&m??q?1K???&?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??'?8??!???V?J@)b??BW"p?1?O{?a?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??(yu???!?wKH4@)??G??V?1>ŜOb??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9'I5(x??I?m???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?^??????^?????!?^?????      ??!       "      ??!       *      ??!       2	7o??#@7o??#@!7o??#@:      ??!       B      ??!       J	??q?j?????q?j???!??q?j???R      ??!       Z	??q?j?????q?j???!??q?j???b      ??!       JCPU_ONLYY'I5(x??b q?m???X@Y      Y@q?FE00???"?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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