	??=??1@??=??1@!??=??1@	???j??????j???!???j???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??=??1@n3??x??A3T?T?}1@Y?M?G????rEagerKernelExecute 0*	????x?V@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeattB??K8??!?!U=!?E@)??ܵ?|??1A???A@:Preprocessing2U
Iterator::Model::ParallelMapV2??+,???!?
?G]?0@)??+,???1?
?G]?0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateϞ??$x??!I?|???4@)?,z????1=~Bߌ?+@:Preprocessing2F
Iterator::Model??Z?[!??!?"<?=@)T?????1%0ΔK*@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??[X7?}?!K???c?@)??[X7?}?1K???c?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliced!:?z?!?\m7??@)d!:?z?1?\m7??@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?@?v??!N?p;??Q@)s?m?B<r?1?W???g@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap$?@?ؔ?!w???.6@)E???V	V?1??R?r??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???j???Iv???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	n3??x??n3??x??!n3??x??      ??!       "      ??!       *      ??!       2	3T?T?}1@3T?T?}1@!3T?T?}1@:      ??!       B      ??!       J	?M?G?????M?G????!?M?G????R      ??!       Z	?M?G?????M?G????!?M?G????b      ??!       JCPU_ONLYY???j???b qv???X@