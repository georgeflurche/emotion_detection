	?s)?*'@?s)?*'@!?s)?*'@	̊tCA??̊tCA??!̊tCA??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?s)?*'@???????A?h????&@YGN??;??rEagerKernelExecute 0*	I+??_@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatd?w?W??!?U?}9n@@)`?????1HyP?r?;@:Preprocessing2F
Iterator::Model4.???!??????@@)??/g???1VVYe?U5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??+,???! kYDk8@)MJA??4??1?7k??1@:Preprocessing2U
Iterator::Model::ParallelMapV2yZ~?*O??!??|?t)@)yZ~?*O??1??|?t)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?U-?(??!*͸?L@)?U-?(??1*͸?L@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	?`??w??!???P@)!>???@??1d?A@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorcz?({?!Kɘ1 ?@)cz?({?1Kɘ1 ?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap%[]N	???!?|??,?:@)W	?3?j?1????F?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9̊tCA??I?y?}?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????????????!???????      ??!       "      ??!       *      ??!       2	?h????&@?h????&@!?h????&@:      ??!       B      ??!       J	GN??;??GN??;??!GN??;??R      ??!       Z	GN??;??GN??;??!GN??;??b      ??!       JCPU_ONLYY̊tCA??b q?y?}?X@