	L???'@L???'@!L???'@	,x??Z??,x??Z??!,x??Z??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:L???'@9'0????A?ډ??X'@Y?~j?t???rEagerKernelExecute 0*??MbXW@)      =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????!????6@@)"??pә?1>???;@:Preprocessing2U
Iterator::Model::ParallelMapV2???,????!?Q???1@)???,????1?Q???1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate1?Zd??!?K?Nʥ<@)?`?????1??'j?0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice@OI???!?u?N??'@)@OI???1?u?N??'@:Preprocessing2F
Iterator::Model?~2Ƈٛ?!(~?W =@)n½2oՅ?1#Y?ϥ?&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???`???!v >??Q@)?\?	?}?1?·f?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensors?<G??t?!o?C3?@)s?<G??t?1o?C3?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????oa??!?
?5?>@)I?V?_?1e@?[? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9-x??Z??I??J?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	9'0????9'0????!9'0????      ??!       "      ??!       *      ??!       2	?ډ??X'@?ډ??X'@!?ډ??X'@:      ??!       B      ??!       J	?~j?t????~j?t???!?~j?t???R      ??!       Z	?~j?t????~j?t???!?~j?t???b      ??!       JCPU_ONLYY-x??Z??b q??J?X@