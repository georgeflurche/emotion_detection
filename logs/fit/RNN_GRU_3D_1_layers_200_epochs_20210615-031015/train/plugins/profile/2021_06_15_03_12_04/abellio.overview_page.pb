?	?ikD0&(@?ikD0&(@!?ikD0&(@		<G?
??	<G?
??!	<G?
??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?ikD0&(@X S??Aݚt["?'@YB?f??j??rEagerKernelExecute 0*	$??C?]@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?K?Ƽ??!???涜B@)?h[?:??1?)/?Q{?@:Preprocessing2U
Iterator::Model::ParallelMapV2?'????!??m?F/@)?'????1??m?F/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?V'g(??!$|4??8@)???Ü??1??^2+@:Preprocessing2F
Iterator::Model??h㈵??!?YI??Z;@)r?????1??$?Qo'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicev?!H??!J6o?-&@)v?!H??1J6o?-&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?լ3?/??!????D)R@)???<????1?? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorEKO?|?!????p?@)EKO?|?1????p?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???R????!Eɹ\?\;@)??{?&j?1?)?A?g@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9	<G?
??I?q????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	X S??X S??!X S??      ??!       "      ??!       *      ??!       2	ݚt["?'@ݚt["?'@!ݚt["?'@:      ??!       B      ??!       J	B?f??j??B?f??j??!B?f??j??R      ??!       Z	B?f??j??B?f??j??!B?f??j??b      ??!       JCPU_ONLYY	<G?
??b q?q????X@Y      Y@qDkx)wI??"?
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