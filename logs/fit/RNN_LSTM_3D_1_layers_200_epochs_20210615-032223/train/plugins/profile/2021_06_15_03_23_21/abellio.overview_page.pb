?	?։???*@?։???*@!?։???*@	Zb0?@???Zb0?@???!Zb0?@???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?։???*@??)??z??A??v??j*@Y~:3P??rEagerKernelExecute 0*	m????"Z@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?@??L??!Kk,2?A@)??6?h???1???˷;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?C6?.6??!??f?[I;@)??9#J{??1?u??22@:Preprocessing2U
Iterator::Model::ParallelMapV2????vܐ?!?z? ?/@)????vܐ?1?z? ?/@:Preprocessing2F
Iterator::Model2??%䃞?!?Y1?<@)f/?N??1??9?%?)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice:???u??!l=?z-"@):???u??1l=?z-"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???ɍ"??!<??3??Q@)Na?????1?(ƒN?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?b*???{?!dtJ-?@)?b*???{?1dtJ-?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap"nN%??!Lԛ??=@)?0e??f?1???9@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Zb0?@???I;??~{?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??)??z????)??z??!??)??z??      ??!       "      ??!       *      ??!       2	??v??j*@??v??j*@!??v??j*@:      ??!       B      ??!       J	~:3P??~:3P??!~:3P??R      ??!       Z	~:3P??~:3P??!~:3P??b      ??!       JCPU_ONLYYZb0?@???b q;??~{?X@Y      Y@q?m(???"?
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