	"????&@"????&@!"????&@	b??????b??????!b??????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:"????&@:???u??A8M?p?&@Y?o?^}<??rEagerKernelExecute 0*	?G?z?^@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat8?GnM???!*??IAA@)2???4??1^?_ի<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate0?GĔ??!?n??U:@)?????>??1$҂? ?0@:Preprocessing2U
Iterator::Model::ParallelMapV2?y?ؘב?!?-??V,@)?y?ؘב?1?-??V,@:Preprocessing2F
Iterator::Modelˀ??,'??!雇?y>;@)???P?v??1
~C&*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?t ??Շ?!?9??"@)?t ??Շ?1?9??"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??	?_???!^?a0R@))_?BF??1??Wֈ{"@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor)????h}?!???G?Z@))????h}?1???G?Z@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????B??! ?
?/=@)??KU??j?1"?) ?Y@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9a??????I;?, ??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	:???u??:???u??!:???u??      ??!       "      ??!       *      ??!       2	8M?p?&@8M?p?&@!8M?p?&@:      ??!       B      ??!       J	?o?^}<???o?^}<??!?o?^}<??R      ??!       Z	?o?^}<???o?^}<??!?o?^}<??b      ??!       JCPU_ONLYYa??????b q;?, ??X@