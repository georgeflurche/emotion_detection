	)H4??.@)H4??.@!)H4??.@	??d'5????d'5??!??d'5??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:)H4??.@?^?sa???A??:?2.@Y????6???rEagerKernelExecute 0*	H?z??Q@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat@?իȘ?!o]?s?@@)A?v??1AnLT??9@:Preprocessing2U
Iterator::Model::ParallelMapV2?>:u峌?!$S??҈3@)?>:u峌?1$S??҈3@:Preprocessing2F
Iterator::Model??u?T??!Ð?D?A@)YİØ???1b??M0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate"T?????!??xYͲ6@)?£?#??1>??n".@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?D???v?!p2?N??@)?D???v?1p2?N??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicew-!?lv?!ޥ?W?@)w-!?lv?1ޥ?W?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipA?+????!????]
P@)/?:?p?1??͈<?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap0???"??!l??U??8@)?Д?~PW?1?L{????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??d'5??IU?????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?^?sa????^?sa???!?^?sa???      ??!       "      ??!       *      ??!       2	??:?2.@??:?2.@!??:?2.@:      ??!       B      ??!       J	????6???????6???!????6???R      ??!       Z	????6???????6???!????6???b      ??!       JCPU_ONLYY??d'5??b qU?????X@