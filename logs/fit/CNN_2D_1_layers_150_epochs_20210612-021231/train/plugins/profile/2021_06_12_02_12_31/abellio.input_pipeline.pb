	غ??0@غ??0@!غ??0@	0&\?i??0&\?i??!0&\?i??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:غ??0@8h?>???A`[??g?/@Y?W???T??rEagerKernelExecute 0*	:??v?oW@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??c?g^??!$?w???@)j???늙?1G?mb??:@:Preprocessing2F
Iterator::ModelcG?P???!?A???C@)?Y.????1??Ho??4@:Preprocessing2U
Iterator::Model::ParallelMapV25???#??!̕e??2@)5???#??1̕e??2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??Or?M??!8?I?R&5@)????????1p?'*?)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceiUMu?! ???{b @)iUMu?1 ???{b @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??_w????!G?(<?(N@)??????u?1?5????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor&U?M?Ms?!w?'?@)&U?M?Ms?1w?'?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??6o???!`?Dc 7@)?lt?Oq\?1?|?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no90&\?i??Iڣ?y??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	8h?>???8h?>???!8h?>???      ??!       "      ??!       *      ??!       2	`[??g?/@`[??g?/@!`[??g?/@:      ??!       B      ??!       J	?W???T???W???T??!?W???T??R      ??!       Z	?W???T???W???T??!?W???T??b      ??!       JCPU_ONLYY0&\?i??b qڣ?y??X@