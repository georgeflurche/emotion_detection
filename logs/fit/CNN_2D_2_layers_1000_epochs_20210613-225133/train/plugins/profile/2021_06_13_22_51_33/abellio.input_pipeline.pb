	?Qd???1@?Qd???1@!?Qd???1@	jTOg???jTOg???!jTOg???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?Qd???1@???????A#ظ?]?1@Y?lY?.ó?rEagerKernelExecute 0*	??S㥛W@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat{m??]??!]h????A@)??E|'f??1? \?g>@:Preprocessing2U
Iterator::Model::ParallelMapV2?K?e?%??!?R?͛?2@)?K?e?%??1?R?͛?2@:Preprocessing2F
Iterator::Model!??nJ??!?"??A@)n2??n??1?7?v??0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??y7??!?o?:?2@)???6T???1_-??17$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?^?????!F??Cu!@)?^?????1F??Cu!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip5
If???!?n7P@)?e???y?1??+ﭽ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?\QJVu?!?ꢆ@)?\QJVu?1?ꢆ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?W??V???!????0?5@)????e?11f?~?_@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9kTOg???I????J?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????????????!???????      ??!       "      ??!       *      ??!       2	#ظ?]?1@#ظ?]?1@!#ظ?]?1@:      ??!       B      ??!       J	?lY?.ó??lY?.ó?!?lY?.ó?R      ??!       Z	?lY?.ó??lY?.ó?!?lY?.ó?b      ??!       JCPU_ONLYYkTOg???b q????J?X@