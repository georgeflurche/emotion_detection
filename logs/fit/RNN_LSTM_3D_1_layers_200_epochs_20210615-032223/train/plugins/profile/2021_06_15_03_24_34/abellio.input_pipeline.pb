	?46?(@?46?(@!?46?(@	?TӪ͊???TӪ͊??!?TӪ͊??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?46?(@?v?k??A&?<Y?(@Ys?????rEagerKernelExecute 0*	????KOY@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???"[??!??X?׽@@)?
?b?0??1?(2?z(<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatejm?kA??!???>B&>@)T??Yh???1?*?*4@:Preprocessing2U
Iterator::Model::ParallelMapV2?&jin???!#r????/@)?&jin???1#r????/@:Preprocessing2F
Iterator::Model??????!w!?K$;@)??]?????1??H??$&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice,?)???!?o??v?#@),?)???1?o??v?#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?"N'????!???v?R@)?*l? {?1?iH??*@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorXr??v?! ]???L@)Xr??v?1 ]???L@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapJ+???!?	??{@@)OYM?]g?1`???i?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?TӪ͊??IWY?d??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?v?k???v?k??!?v?k??      ??!       "      ??!       *      ??!       2	&?<Y?(@&?<Y?(@!&?<Y?(@:      ??!       B      ??!       J	s?????s?????!s?????R      ??!       Z	s?????s?????!s?????b      ??!       JCPU_ONLYY?TӪ͊??b qWY?d??X@