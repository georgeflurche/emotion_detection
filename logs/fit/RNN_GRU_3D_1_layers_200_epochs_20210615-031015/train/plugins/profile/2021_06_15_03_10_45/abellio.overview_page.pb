?	???"1?&@???"1?&@!???"1?&@	???w?????w??!???w??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:???"1?&@??Y?w??AÁ?,`J&@Y????kz??rEagerKernelExecute 0*	? ?rh?d@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?'?????!R?Z?dfG@)????㾭?1????mA@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeaty??.??!Q?&׮>@)؟??N???1???2=?2@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?nJy???!m#??3;(@)?nJy???1m#??3;(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?=Զa??!Ԅ?Vl?'@)?=Զa??1Ԅ?Vl?'@:Preprocessing2U
Iterator::Model::ParallelMapV22r??Ï?!?ǥ???"@)2r??Ï?1?ǥ???"@:Preprocessing2F
Iterator::Model=?q??!i??Gh?.@)G6u??1?f???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??^EF??!???r U@)QL? 3?1????G@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapv?Kp???!&=1#??H@)T????p?1Hmd???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???w??I???Y?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??Y?w????Y?w??!??Y?w??      ??!       "      ??!       *      ??!       2	Á?,`J&@Á?,`J&@!Á?,`J&@:      ??!       B      ??!       J	????kz??????kz??!????kz??R      ??!       Z	????kz??????kz??!????kz??b      ??!       JCPU_ONLYY???w??b q???Y?X@Y      Y@q?L0?>O??"?
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