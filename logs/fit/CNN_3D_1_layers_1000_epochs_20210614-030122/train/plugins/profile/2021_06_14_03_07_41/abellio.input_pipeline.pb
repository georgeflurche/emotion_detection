	?d73??&@?d73??&@!?d73??&@	?ޭ??:???ޭ??:??!?ޭ??:??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?d73??&@?f?\S ??A\:?<C&@YN?G????rEagerKernelExecute 0*	??ʡ?X@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?P?f???!kDn cB@)'???S??1~rr?8?=@:Preprocessing2U
Iterator::Model::ParallelMapV2?[Ɏ???!??U?E0@)?[Ɏ???1??U?E0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???'??!?@??H?7@)??e6??1??BCΉ+@:Preprocessing2F
Iterator::Model=?U????!??? +=@)|~!<??16m??m?)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice. ??L??!5?????#@). ??L??15?????#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?3??O??!??=?Q@)??Cl??1y???% @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?'??Q|?![Y??@)?'??Q|?1[Y??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?7?n??!?QbLl?9@)????8b?1G?X??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?ޭ??:??IC?Lb??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?f?\S ???f?\S ??!?f?\S ??      ??!       "      ??!       *      ??!       2	\:?<C&@\:?<C&@!\:?<C&@:      ??!       B      ??!       J	N?G????N?G????!N?G????R      ??!       Z	N?G????N?G????!N?G????b      ??!       JCPU_ONLYY?ޭ??:??b qC?Lb??X@