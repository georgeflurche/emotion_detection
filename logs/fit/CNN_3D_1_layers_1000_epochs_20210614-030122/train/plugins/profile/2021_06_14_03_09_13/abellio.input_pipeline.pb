	K?h?$@K?h?$@!K?h?$@	?P?\?	???P?\?	??!?P?\?	??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:K?h?$@??5w????Ad??3??$@Y?c?]Kȯ?rEagerKernelExecute 0*	أp=
W\@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??'?Ȥ?!r,:N?A@)@?R??1?I??z?=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?im?k??!A???>@)#?????1?4?/?3@:Preprocessing2U
Iterator::Model::ParallelMapV2?e???-??!,:N?")@)?e???-??1,:N?")@:Preprocessing2F
Iterator::ModelL?'????!.?G?6@)ݖ?g???1?!$xo?$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??W?<ׇ?!???}??$@)??W?<ׇ?1???}??$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipd??Tka??!|?.?GS@)?8*7QK??1?A;?? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??-?{?!m=L
??@)??-?{?1m=L
??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?8ӄ?'??!
l?O??@@){?\?&?k?1?????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?P?\?	??I^YFS??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??5w??????5w????!??5w????      ??!       "      ??!       *      ??!       2	d??3??$@d??3??$@!d??3??$@:      ??!       B      ??!       J	?c?]Kȯ??c?]Kȯ?!?c?]Kȯ?R      ??!       Z	?c?]Kȯ??c?]Kȯ?!?c?]Kȯ?b      ??!       JCPU_ONLYY?P?\?	??b q^YFS??X@