	?s?L0(@?s?L0(@!?s?L0(@	AOz??H??AOz??H??!AOz??H??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?s?L0(@VH?I?O??A? ??*?'@Y:???????rEagerKernelExecute 0*	+????[@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???>$??!{޹3?A@)8??9??1?T??|<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????Ss??!?Q??!?>@)?\????1?????4@:Preprocessing2U
Iterator::Model::ParallelMapV22?CP5??!a???'@)2?CP5??1a???'@:Preprocessing2F
Iterator::Model???????!???lZ?5@)N?»\ć?1L?H?$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice6?????!{??S?V#@)6?????1{??S?V#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?4?Ry;??!?R?d)?S@)????????1?J?"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorКiQ?!?T?9?}@)КiQ?1?T?9?}@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap_^?}t???!??6}ݚ@@)?<֌rg?1??o˔@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9AOz??H??Ia9?n?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	VH?I?O??VH?I?O??!VH?I?O??      ??!       "      ??!       *      ??!       2	? ??*?'@? ??*?'@!? ??*?'@:      ??!       B      ??!       J	:???????:???????!:???????R      ??!       Z	:???????:???????!:???????b      ??!       JCPU_ONLYYAOz??H??b qa9?n?X@