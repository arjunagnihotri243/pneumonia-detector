	"lxz??s@"lxz??s@!"lxz??s@	0?b????0?b????!0?b????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$"lxz??s@?C?l????A?S㥛?s@Y?D???J??*	?????yQ@2U
Iterator::Model::ParallelMapV2ŏ1w-!??!%?XJϾ5@)ŏ1w-!??1%?XJϾ5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?&S???!??:??	:@)ŏ1w-!??1%?XJϾ5@:Preprocessing2F
Iterator::Model?:pΈҞ?!??f?߇E@)2??%䃎?1-2u@?P5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?q??????!p?2NQ6@)??y?):??1?a?w)@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceS?!?uq{?!S?+#@)S?!?uq{?1S?+#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip8??d?`??!X?: xL@)/n??r?1?t??-@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?~j?t?h?!?V???*@)?~j?t?h?1?V???*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap"??u????!?/??H?8@)-C??6Z?1b)=??O@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9/?b????I??A???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?C?l?????C?l????!?C?l????      ??!       "      ??!       *      ??!       2	?S㥛?s@?S㥛?s@!?S㥛?s@:      ??!       B      ??!       J	?D???J???D???J??!?D???J??R      ??!       Z	?D???J???D???J??!?D???J??b      ??!       JCPU_ONLYY/?b????b q??A???X@