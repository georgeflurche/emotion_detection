?	???sEa(@???sEa(@!???sEa(@	???E?????E??!???E??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:???sEa(@!撪?&??A:W???'@Y[z4Փ??rEagerKernelExecute 0*	J+??\@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat7??ͧ?!鎖JND@)zrM??΢?1?qzVU@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate5bf??(??!7?2崔:@)??????1n|?X<?2@:Preprocessing2U
Iterator::Model::ParallelMapV2?~???!?8i??x*@)?~???1?8i??x*@:Preprocessing2F
Iterator::Model?Q??????!?n????7@)?$xC??1???:?$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??A_z???!j?Q ?!@)??A_z???1j?Q ?!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceb??!????!"G?0??@)b??!????1"G?0??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipN?=??j??!^???S@)M??u??1?y???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap y?P????!?_??=o<@)c?D(ba?1"?<????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???E??I??.t??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!撪?&??!撪?&??!!撪?&??      ??!       "      ??!       *      ??!       2	:W???'@:W???'@!:W???'@:      ??!       B      ??!       J	[z4Փ??[z4Փ??![z4Փ??R      ??!       Z	[z4Փ??[z4Փ??![z4Փ??b      ??!       JCPU_ONLYY???E??b q??.t??X@Y      Y@q?o?tn??"?
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