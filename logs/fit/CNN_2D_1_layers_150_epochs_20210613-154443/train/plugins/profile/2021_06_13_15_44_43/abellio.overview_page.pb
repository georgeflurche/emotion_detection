?	??jx+@??jx+@!??jx+@	?Ӽ?????Ӽ????!?Ӽ????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??jx+@lBZc?	??AO??:7-+@Y?Hi6?è?rEagerKernelExecute 0*	?K7?A?N@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?X??L/??!???Nt;@)0? ?????1??R?J6@:Preprocessing2U
Iterator::Model::ParallelMapV20??!???!??{K5@)0??!???1??{K5@:Preprocessing2F
Iterator::ModeldT8?T??!??NE@)??~? ??1?????4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?jGq?:??!莣???4@)y?ѩ+?1|?<?#?(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicexԘsIu?!U?
b!@)xԘsIu?1U?
b!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipX???!??!3????L@)?\??Jr?1??cr9@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?Z(???i?!???
??@)?Z(???i?1???
??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9??U??!?[?B?,7@)r?#DV?1?fN 2?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?Ӽ????I,C?@v?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	lBZc?	??lBZc?	??!lBZc?	??      ??!       "      ??!       *      ??!       2	O??:7-+@O??:7-+@!O??:7-+@:      ??!       B      ??!       J	?Hi6?è??Hi6?è?!?Hi6?è?R      ??!       Z	?Hi6?è??Hi6?è?!?Hi6?è?b      ??!       JCPU_ONLYY?Ӽ????b q,C?@v?X@Y      Y@q?6??X??"?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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