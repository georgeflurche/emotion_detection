	?$?3'@?$?3'@!?$?3'@	iB?;?A??iB?;?A??!iB?;?A??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?$?3'@?ฌ???A?z???&@Y??>s֧??rEagerKernelExecute 0*	?S㥛?Z@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS??????!???;BSA@)"o???I??1?B???;@:Preprocessing2U
Iterator::Model::ParallelMapV2??Or?M??! 榏 ?2@)??Or?M??1 榏 ?2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???????!????6@)??x????1?o??a?*@:Preprocessing2F
Iterator::Modelio???T??!?2?????@)<l"3???1??\ I*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?M????!??i???#@)?M????1??i???#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip(??????!Ys?ГQ@)?(]?????1끜?2!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor^?SH~?!??%C?@)^?SH~?1??%C?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?ɩ?aj??!1'?y?9@)??7???b?1???%5@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9jB?;?A??I{???|?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?ฌ????ฌ???!?ฌ???      ??!       "      ??!       *      ??!       2	?z???&@?z???&@!?z???&@:      ??!       B      ??!       J	??>s֧????>s֧??!??>s֧??R      ??!       Z	??>s֧????>s֧??!??>s֧??b      ??!       JCPU_ONLYYjB?;?A??b q{???|?X@