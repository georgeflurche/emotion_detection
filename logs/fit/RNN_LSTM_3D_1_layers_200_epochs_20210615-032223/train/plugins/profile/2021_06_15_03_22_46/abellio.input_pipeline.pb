	?qQ-"?(@?qQ-"?(@!?qQ-"?(@	???F??????F???!???F???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?qQ-"?(@^?}t?ʿ?A,d??=(@Y?Z	?%q??rEagerKernelExecute 0*	?v??;b@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea?hV???!?R?F@)??-$`??1┭??B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?74e???!;w????:@)f.py???1?h?6@:Preprocessing2U
Iterator::Model::ParallelMapV24??`??!???!%"@)4??`??1???!%"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip㊋?r??!>?d1?U@)? ??=@??16+?V#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?H.?!???!|???"?@)?H.?!???1|???"?@:Preprocessing2F
Iterator::Model?o?N\???!??t??/@)>U?W??1?????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?F ^?/x?!?Hg
?1@)?F ^?/x?1?Hg
?1@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapj?:?z??!xG?NF?H@)?m?v?1??Tc@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???F???I ?Rr??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	^?}t?ʿ?^?}t?ʿ?!^?}t?ʿ?      ??!       "      ??!       *      ??!       2	,d??=(@,d??=(@!,d??=(@:      ??!       B      ??!       J	?Z	?%q???Z	?%q??!?Z	?%q??R      ??!       Z	?Z	?%q???Z	?%q??!?Z	?%q??b      ??!       JCPU_ONLYY???F???b q ?Rr??X@