	???j?(@???j?(@!???j?(@	?mCp]???mCp]??!?mCp]??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:???j?(@k`?????AyGsd}'@Y????岱?rEagerKernelExecute 0*	G?z??\@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?;? Ѥ?!Q#Pj?A@)?rJ_??1??35=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???1>̞?!h?O,`=:@)Ӈ.?o???1(??%۲0@:Preprocessing2U
Iterator::Model::ParallelMapV2?1?=B͐?!?wn}X?,@)?1?=B͐?1?wn}X?,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipt@??$??!????R@)?} R?8??1-??}%@:Preprocessing2F
Iterator::Model???%:˜?!??AI?8@)??????1???:o$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????e??!?@?
#@)????e??1?@?
#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorVG?tF~?!?&J??@)VG?tF~?1?&J??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?% ??*??!kvW^@=@)??bc^Gl?1?e4Y?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?mCp]??I%y?E?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	k`?????k`?????!k`?????      ??!       "      ??!       *      ??!       2	yGsd}'@yGsd}'@!yGsd}'@:      ??!       B      ??!       J	????岱?????岱?!????岱?R      ??!       Z	????岱?????岱?!????岱?b      ??!       JCPU_ONLYY?mCp]??b q%y?E?X@