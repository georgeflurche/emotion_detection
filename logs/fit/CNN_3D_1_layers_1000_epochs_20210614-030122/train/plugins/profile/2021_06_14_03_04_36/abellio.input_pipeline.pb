	}?%??(@}?%??(@!}?%??(@	?1?뫿???1?뫿??!?1?뫿??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:}?%??(@$?@????A??
a5?'@Y?8ӄ?'??rEagerKernelExecute 0*	??ʡE?_@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??%!???!z?`ihA@)?(z?c???1?z+k??<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?8?:V??!G?i?2?:@)??????1?t?I82@:Preprocessing2U
Iterator::Model::ParallelMapV2`cD?В?!??l?-@)`cD?В?1??l?-@:Preprocessing2F
Iterator::Model?f??}q??!??0HB?:@)??מY??1???u??(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipJ|?????!X??m?ER@)?y7R??1uv?K?7!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceA??_???!????!@)A??_???1????!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor#h?$??!
??YM?@)#h?$??1
??YM?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???#G:??!1?OP??=@)|,G?@n?1Q//?<V@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?1?뫿??I??(???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	$?@????$?@????!$?@????      ??!       "      ??!       *      ??!       2	??
a5?'@??
a5?'@!??
a5?'@:      ??!       B      ??!       J	?8ӄ?'???8ӄ?'??!?8ӄ?'??R      ??!       Z	?8ӄ?'???8ӄ?'??!?8ӄ?'??b      ??!       JCPU_ONLYY?1?뫿??b q??(???X@