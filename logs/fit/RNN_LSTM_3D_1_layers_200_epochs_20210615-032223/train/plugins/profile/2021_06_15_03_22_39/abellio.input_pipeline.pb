	H?ξ??%@H?ξ??%@!H?ξ??%@	E?v????E?v????!E?v????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:H?ξ??%@"??u????A6?Ko.%@Y??Kǜ??rEagerKernelExecute 0*	????c]@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateIM??f???!?/??C@))???^??1???$?>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??CR%??![$i?b??@)b??BW"??1ZU????:@:Preprocessing2U
Iterator::Model::ParallelMapV2\;Qi??!Y?]??&@)\;Qi??1Y?]??&@:Preprocessing2F
Iterator::ModelV?F?????!??x$4@)P?<???1??>??!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice~?*O ???!?yQ8a!@)~?*O ???1?yQ8a!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?ݮ????!B?!???S@)F?̱??~?1nDSEz@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?*5{?x?! <;??@)?*5{?x?1 <;??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap!??F???!???fF?D@)?T?:?e?1????d@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9F?v????I???v?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	"??u????"??u????!"??u????      ??!       "      ??!       *      ??!       2	6?Ko.%@6?Ko.%@!6?Ko.%@:      ??!       B      ??!       J	??Kǜ????Kǜ??!??Kǜ??R      ??!       Z	??Kǜ????Kǜ??!??Kǜ??b      ??!       JCPU_ONLYYF?v????b q???v?X@