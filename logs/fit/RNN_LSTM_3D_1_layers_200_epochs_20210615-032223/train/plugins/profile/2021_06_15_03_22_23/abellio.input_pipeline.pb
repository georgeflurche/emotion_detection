	?w???<,@?w???<,@!?w???<,@	?MW??????MW?????!?MW?????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?w???<,@3??(???A\?	???+@Y~!<?8??rEagerKernelExecute 0*	??????f@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?? Z+ڴ?!? ?'PF@)?:????1oN'Ӳ>@:Preprocessing2F
Iterator::Model?> ?M???!?Y?7?A@)???0a??1?a?P??5@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????M??!Zf?97-@)????M??1Zf?97-@:Preprocessing2U
Iterator::Model::ParallelMapV2?? kծ??!????u{+@)?? kծ??1????u{+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateͭVc	??!N?V?J?,@)???g????1?I?n?"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?M???P??!? ?O?@)?M???P??1? ?O?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???????!әy?P@)?J?E?}?1?K?'?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap|H??ߠ??!C??.C?/@)s?<G??d?1????/??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?MW?????IdQ??|?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	3??(???3??(???!3??(???      ??!       "      ??!       *      ??!       2	\?	???+@\?	???+@!\?	???+@:      ??!       B      ??!       J	~!<?8??~!<?8??!~!<?8??R      ??!       Z	~!<?8??~!<?8??!~!<?8??b      ??!       JCPU_ONLYY?MW?????b qdQ??|?X@