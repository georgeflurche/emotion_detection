	j.7?(@j.7?(@!j.7?(@	Ր??E??Ր??E??!Ր??E??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:j.7?(@??q4GV??A????X}'@Yc?tv28??rEagerKernelExecute 0*	S???f@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate1??c???!u՜u?N@)$'?
b??1m???,B@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?@1?d??!4?ѓ?8@)?@1?d??14?ѓ?8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatEg?E(???!:?za??4@)?r߉Y??1$??Œc1@:Preprocessing2U
Iterator::Model::ParallelMapV2?????!??!??^??@)?????!??1??^??@:Preprocessing2F
Iterator::Model\?O???!?v???`'@)
?\????1?5@ ?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?)????!(&??V@)??Co???1:??Ͻ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorin??Kx?!??9?,?
@)in??Kx?1??9?,?
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???FXT??!??gmO@)^0????g?1?B???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Ր??E??I+oB??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??q4GV????q4GV??!??q4GV??      ??!       "      ??!       *      ??!       2	????X}'@????X}'@!????X}'@:      ??!       B      ??!       J	c?tv28??c?tv28??!c?tv28??R      ??!       Z	c?tv28??c?tv28??!c?tv28??b      ??!       JCPU_ONLYYՐ??E??b q+oB??X@