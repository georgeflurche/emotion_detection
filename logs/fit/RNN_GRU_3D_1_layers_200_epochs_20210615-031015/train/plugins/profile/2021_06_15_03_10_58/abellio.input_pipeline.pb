	?f??%@?f??%@!?f??%@	~ir(?m??~ir(?m??!~ir(?m??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?f??%@?EИI??A>U?WS%@Y}$%=???rEagerKernelExecute 0*	?(\??Eb@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?????L??!o??YG@)???P1ί?1?qKn?E@:Preprocessing2U
Iterator::Model::ParallelMapV2
3?`??!?ߕ?4?3@)
3?`??1?ߕ?4?3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??% ????!B??:?1@)'??0???1???=?&@:Preprocessing2F
Iterator::Model?M?ɤ?!?U
}?;@)\?-??e??1?y~??L @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice9??!??!ԙ8pl?@)9??!??1ԙ8pl?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?U-?(??!ݸj?`R@)7?',????1?????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor+l? [v?!h%$ع?@)+l? [v?1h%$ع?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap? ?> ???!??k??3@)l#?	?h?1?S\?qs @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9~ir(?m??I-?{$?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?EИI???EИI??!?EИI??      ??!       "      ??!       *      ??!       2	>U?WS%@>U?WS%@!>U?WS%@:      ??!       B      ??!       J	}$%=???}$%=???!}$%=???R      ??!       Z	}$%=???}$%=???!}$%=???b      ??!       JCPU_ONLYY~ir(?m??b q-?{$?X@