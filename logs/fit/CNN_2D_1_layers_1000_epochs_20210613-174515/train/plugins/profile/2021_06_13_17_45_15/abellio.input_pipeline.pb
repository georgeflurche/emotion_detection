	????
-@????
-@!????
-@	??"?fr????"?fr??!??"?fr??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:????
-@?3iSu???A?B?i??,@Y???5????rEagerKernelExecute 0*	-???OS@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatܜJ?*??!??6?KC@).??H??1??ʳ???@:Preprocessing2U
Iterator::Model::ParallelMapV2Nё\?C??!Pc?&?0@)Nё\?C??1Pc?&?0@:Preprocessing2F
Iterator::Model??U?@ؙ?!I?b?V@@)????l??1?*b??0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate1???z??![??7|?4@)?F;n?݄?1?If/a*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?F ^?/x?!?????@)?F ^?/x?1?????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(??я?s?!4'?z?@)(??я?s?14'?z?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZippB!???!\?N???P@)?!p$?`s?1???U@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????g???!?? yy7@)??bc^G\?1??_??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??"?fr??Ik????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?3iSu????3iSu???!?3iSu???      ??!       "      ??!       *      ??!       2	?B?i??,@?B?i??,@!?B?i??,@:      ??!       B      ??!       J	???5???????5????!???5????R      ??!       Z	???5???????5????!???5????b      ??!       JCPU_ONLYY??"?fr??b qk????X@