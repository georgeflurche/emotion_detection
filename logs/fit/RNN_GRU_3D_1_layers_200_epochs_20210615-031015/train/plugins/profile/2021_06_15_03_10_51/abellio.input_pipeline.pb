	??X???&@??X???&@!??X???&@	??!MK????!MK??!??!MK??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??X???&@??l?%???A&S??j&@Y?????%??rEagerKernelExecute 0*	?A`??b@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???҈???!ħ?<J?G@)?Z
H???1??؋??E@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateJ{?/L???!?-?ƙG9@)G仔?d??1?qai>3@:Preprocessing2U
Iterator::Model::ParallelMapV23???/??!l?Â0?%@)3???/??1l?Â0?%@:Preprocessing2F
Iterator::Modelx$(~??!???B?O3@)???o^???1???? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip ?viý?!LO?,T@)"?*??<??1?f???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?$???ρ?!??$@)?$???ρ?1??$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorm?y?ؘw?!?>mk?@)m?y?ؘw?1?>mk?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapo???ģ?!2G}V??:@)Tb.?a?1??Q?x@??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??!MK??I???ei?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??l?%?????l?%???!??l?%???      ??!       "      ??!       *      ??!       2	&S??j&@&S??j&@!&S??j&@:      ??!       B      ??!       J	?????%???????%??!?????%??R      ??!       Z	?????%???????%??!?????%??b      ??!       JCPU_ONLYY??!MK??b q???ei?X@