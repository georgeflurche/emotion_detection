?	w???s?*@w???s?*@!w???s?*@	?0 MZY???0 MZY??!?0 MZY??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:w???s?*@E?4f??A?2?G*@YW??Ma???rEagerKernelExecute 0*V-*Z@)      =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??HP??!RktwT?B@)?}s???1{Q????@:Preprocessing2U
Iterator::Model::ParallelMapV2??^??W??!?
8?.0@)??^??W??1?
8?.0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????ّ??!???̼?8@)?.ޏ?/??1'??-@:Preprocessing2F
Iterator::Model1[?*?!??`o?;@)????wԈ?1???'+'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice,??????!?????{$@),??????1?????{$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip,???cZ??!Z?g8?R@)ҏ?S??{?1x??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor}?E?z?!?"A?
@)}?E?z?1?"A?
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???#Fϝ?!fIs??;@)?ɐc?i?1??C?i/@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?0 MZY??I??eKM?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	E?4f??E?4f??!E?4f??      ??!       "      ??!       *      ??!       2	?2?G*@?2?G*@!?2?G*@:      ??!       B      ??!       J	W??Ma???W??Ma???!W??Ma???R      ??!       Z	W??Ma???W??Ma???!W??Ma???b      ??!       JCPU_ONLYY?0 MZY??b q??eKM?X@Y      Y@q6?Z?u??"?
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