	?=?N?G'@?=?N?G'@!?=?N?G'@	?^?C ????^?C ???!?^?C ???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?=?N?G'@?R	O???A[??8??&@Y?1??l??rEagerKernelExecute 0*	?O??n\@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?b??^'??!p?{`B@)??	????1????>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?4????!?@"? :@)?_w??ē?1???k?,1@:Preprocessing2U
Iterator::Model::ParallelMapV2n??t???!?Rq??-@)n??t???1?Rq??-@:Preprocessing2F
Iterator::Model?\??u??!*4???;@)X?\T??1na?j(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicejM??S??!?>y?!@)jM??S??1?>y?!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?B?????!?r?C>R@)?d??)??1?L??l?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??5?e|?!?`?at?@)??5?e|?1?`?at?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????N???!?m?#=?<@)??^
j?1og??P?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?^?C ???IB?x???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?R	O????R	O???!?R	O???      ??!       "      ??!       *      ??!       2	[??8??&@[??8??&@![??8??&@:      ??!       B      ??!       J	?1??l???1??l??!?1??l??R      ??!       Z	?1??l???1??l??!?1??l??b      ??!       JCPU_ONLYY?^?C ???b qB?x???X@