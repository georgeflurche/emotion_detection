	)?????&@)?????&@!)?????&@	Ws??)??Ws??)??!Ws??)??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:)?????&@]?gA(???A@?:s!&@Y"?4???rEagerKernelExecute 0*	?V?\@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat%vmo?$??!k?$??C@),??26t??1?Qck@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?g??????!??????7@)!V?a???1?@?7?/@:Preprocessing2U
Iterator::Model::ParallelMapV2A?m??!?N???,@)A?m??1?N???,@:Preprocessing2F
Iterator::Model (??{ԟ?!?v?uw?:@)?1 Ǟ??1      )@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?g?????!`"?"?HR@)D??]L??1??.??I @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicet#,*?t??!?!??'@)t#,*?t??1?!??'@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?\?	?}?!???n?@)?\?	?}?1???n?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??,`???!aӦQ??9@)1E?4~?e?1Lm???w@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Vs??)??I?
ڭ?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	]?gA(???]?gA(???!]?gA(???      ??!       "      ??!       *      ??!       2	@?:s!&@@?:s!&@!@?:s!&@:      ??!       B      ??!       J	"?4???"?4???!"?4???R      ??!       Z	"?4???"?4???!"?4???b      ??!       JCPU_ONLYYVs??)??b q?
ڭ?X@