	@?#H? .@@?#H? .@!@?#H? .@	5?>??b??5?>??b??!5?>??b??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:@?#H? .@V)=?K???AʣaQ?-@Y<hv?[???rEagerKernelExecute 0*	?p=
?Cb@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??)x
??!????M?@@)?A?Ѫ???1?1e??;@:Preprocessing2F
Iterator::Model<???ܴ??!bPA?..A@)噗????1<cT?2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?tp?x??!?^??xZ7@)?g$B#ؘ?1H%?J??0@:Preprocessing2U
Iterator::Model::ParallelMapV2^?c@?z??!{??b/@)^?c@?z??1{??b/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??;?2??!??:?@)??;?2??1??:?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipp@KW????!?Wߩ?hP@)]3?f???1=?Q#h?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??\5ρ?!?-q}?@)??\5ρ?1?-q}?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??ѫJ??!?f?Rm?9@)?ZӼ?m?1FC@??o@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.6% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no95?>??b??I???:?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	V)=?K???V)=?K???!V)=?K???      ??!       "      ??!       *      ??!       2	ʣaQ?-@ʣaQ?-@!ʣaQ?-@:      ??!       B      ??!       J	<hv?[???<hv?[???!<hv?[???R      ??!       Z	<hv?[???<hv?[???!<hv?[???b      ??!       JCPU_ONLYY5?>??b??b q???:?X@