?	?T??1*@?T??1*@!?T??1*@	?#?@???#?@??!?#?@??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?T??1*@????1??A?Ry=?)@Y??hUM??rEagerKernelExecute 0*	??v???X@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatSςP?ǡ?!?-??A@)࢓?????1y??«?=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?EaE??!i?ۛ¼;@) R?8?ߑ?1??A*?1@:Preprocessing2U
Iterator::Model::ParallelMapV2???)???!??K B.@)???)???1??K B.@:Preprocessing2F
Iterator::Model#k??"??!?j??9@)??t 멅?1I5K7j%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice)狽_??!?l;?0#$@))狽_??1?l;?0#$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?C?H????!?S%y?R@)|C??up??14 l?@ @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Ov?! #;??@)????Ov?1 #;??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap%Ί??>??!??:ߩ?>@)|DL?$zi?1???:/	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?#?@??I?g????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????1??????1??!????1??      ??!       "      ??!       *      ??!       2	?Ry=?)@?Ry=?)@!?Ry=?)@:      ??!       B      ??!       J	??hUM????hUM??!??hUM??R      ??!       Z	??hUM????hUM??!??hUM??b      ??!       JCPU_ONLYY?#?@??b q?g????X@Y      Y@qN??W???"?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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