"?6
BHostIDLE"IDLE1?O???[?@A?O???[?@a???N??i???N???Unknown
?HostMaxPoolGrad":gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGrad(1?Q??(?@9?Q??(?@A?Q??(?@I?Q??(?@a??(=׵?i???N????Unknown
pHost_FusedConv2D"sequential/conv2d/Relu(1?v??Z??@9?v??Z??@A?v??Z??@I?v??Z??@a?N??????i?????Y???Unknown
?HostConv2DBackpropFilter";gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilter(1??C?l|?@9??C?l|?@A??C?l|?@I??C?l|?@af???<??i?h??a???Unknown
^HostGatherV2"GatherV2(1?&1?M?@9?&1?M?@A?&1?M?@I?&1?M?@a?"??o%??i?>???????Unknown
~HostReluGrad"(gradient_tape/sequential/conv2d/ReluGrad(1??"????@9??"????@A??"????@I??"????@aP???;???i`v??p????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1B`??"*?@9B`??"*?@AB`??"*?@IB`??"*?@a|?ӊuh??i??G?????Unknown
{HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1h??|?&?@9h??|?&?@Ah??|?&?@Ih??|?&?@aɡDxV??i?9???]???Unknown
?	HostBiasAddGrad"3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGrad(1w??/??@9w??/??@Aw??/??@Iw??/??@a??p?.???i??q ???Unknown
u
HostMaxPool" sequential/max_pooling2d/MaxPool(1????K?@9????K?@A????K?@I????K?@a??N+?U??i.7?ά???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1J+?.w@9J+?.w@AJ+?.w@IJ+?.w@af??Rwm??i?I_?????Unknown
qHostCast"sequential/dropout/dropout/Cast(1??~j?li@9??~j?li@A??~j?li@I??~j?li@aмG??|?i:???|P???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1??"??Nd@9??"??Nd@A??"??Nd@I??"??Nd@aXo???&w?i???~???Unknown
oHostMul"sequential/dropout/dropout/Mul(1bX9??c@9bX9??c@AbX9??c@IbX9??c@a?d`?T@v?i??HK????Unknown
HostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1??C?lkb@9??C?lkb@A??C?lkb@I??C?lkb@a	k???t?i??J????Unknown
}HostMul",gradient_tape/sequential/dropout/dropout/Mul(1??? ??a@9??? ??a@A??? ??a@I??? ??a@aڝ????t?iD?-R????Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1???K7?`@9???K7?`@A???K7?`@I???K7?`@a???s?irNa$???Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1?? ?r?_@9?? ?r?_@A?? ?r?_@I?? ?r?_@a????]?q?i???SH???Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(1?S㥛T]@9?S㥛T]@A?S㥛T]@I?S㥛T]@aS?????p?iP????i???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1F????XJ@9F????XJ@AF????XJ@IF????XJ@a???:	^?i???e?x???Unknown?
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1u?V>G@9u?V>G@Au?V>G@Iu?V>G@a???|Z?i?D($????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1?ZdKD@9?ZdKD@A??v???@@I??v???@@a~~??HS?i????????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1T㥛?0@@9T㥛?0@@AT㥛?0@@IT㥛?0@@a?d??uR?ixQg0?????Unknown
iHostWriteSummary"WriteSummary(1??C?l?:@9??C?l?:@A??C?l?:@I??C?l?:@a;? ?W>N?i?QT?v????Unknown?
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1h??|??8@9h??|??8@Ah??|??8@Ih??|??8@a?>??_OL?i?NI??????Unknown
mHostSoftmax"sequential/dense/Softmax(1o??ʡ7@9o??ʡ7@Ao??ʡ7@Io??ʡ7@a?(?|??J?i?{??F????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1???SA@9???SA@A??(\?b6@I??(\?b6@a??b??I?i?'?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(133333?0@933333?0@A33333?0@I33333?0@a??$?H.C?i??'?s????Unknown
gHostStridedSlice"strided_slice(1?/?$?-@9?/?$?-@A?/?$?-@I?/?$?-@a?_[5?@?i??~??????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1X9??v~,@9X9??v~,@AX9??v~,@IX9??v~,@a?:+??=@?iUӣ+?????Unknown
ZHostArgMax"ArgMax(1?V-)@9?V-)@A?V-)@I?V-)@ae?z?<?i?R??N????Unknown
? HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1+????'@9+????'@A+????'@I+????'@a3x{??R;?iQb???????Unknown
d!HostDataset"Iterator::Model(1V-???9@9V-???9@AT㥛?`'@IT㥛?`'@a??8???:?igIS?????Unknown
?"HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1?x?&1H'@9?x?&1H'@A?x?&1H'@I?x?&1H'@aV+m8??:?iWz'_????Unknown
V#HostSum"Sum_2(1?????&@9?????&@A?????&@I?????&@ab]9?+#9?i8????????Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_2(133333?%@933333?%@A33333?%@I33333?%@a?_???8?i??-?????Unknown
Y%HostPow"Adam/Pow(1???Mb%@9???Mb%@A???Mb%@I???Mb%@a???i8?i?????????Unknown
v&HostAssignAddVariableOp"AssignAddVariableOp_3(19??v?_#@99??v?_#@A9??v?_#@I9??v?_#@afR??16?i?7a^????Unknown
x'HostDataset"#Iterator::Model::ParallelMapV2::Zip(1B`??"?U@9B`??"?U@Ad;?O??"@Id;?O??"@aH(?z5?i??????Unknown
[(HostAddV2"Adam/add(1?K7?A?!@9?K7?A?!@A?K7?A?!@I?K7?A?!@a??Ha4?i??}ߙ????Unknown
l)HostIteratorGetNext"IteratorGetNext(1Zd;ߏ!@9Zd;ߏ!@AZd;ߏ!@IZd;ߏ!@a ??}_4?i??m?????Unknown
t*HostAssignAddVariableOp"AssignAddVariableOp(1?Q??k!@9?Q??k!@A?Q??k!@I?Q??k!@a??\??3?i?9	?????Unknown
e+Host
LogicalAnd"
LogicalAnd(1????K7 @9????K7 @A????K7 @I????K7 @a?s???|2?i?M՚?????Unknown?
?,HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1B`??"[@9B`??"[@AB`??"[@IB`??"[@a??4???1?i?t?!????Unknown
{-HostSum"*categorical_crossentropy/weighted_loss/Sum(1?z?G?@9?z?G?@A?z?G?@I?z?G?@a?Z?8?1?i$???T????Unknown
b.HostDivNoNan"div_no_nan_1(1!?rh?m@9!?rh?m@A!?rh?m@I!?rh?m@aAE?G40?i?vW[????Unknown
?/HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1?Zd;@9?Zd;@A?Zd;@I?Zd;@a5?_?/?i???L????Unknown
?0HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1J+?@9J+?@AJ+?@IJ+?@a??Q?{?.?iY':????Unknown
?1HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1?/?$@9?/?$@A?/?$@I?/?$@a??<??.?i??<'????Unknown
\2HostArgMax"ArgMax_1(1?$???@9?$???@A?$???@I?$???@a???_-?ik???????Unknown
`3HostGatherV2"
GatherV2_1(1Zd;?O?@9Zd;?O?@AZd;?O?@IZd;?O?@a????5!-?i0??????Unknown
t4HostReadVariableOp"Adam/Cast/ReadVariableOp(1?n??J@9?n??J@A?n??J@I?n??J@aWj?|?+?i???0?????Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_4(1?5^?I?@9?5^?I?@A?5^?I?@I?5^?I?@a+i?l'?i;-?? ????Unknown
[6HostPow"
Adam/Pow_1(1\???(?@9\???(?@A\???(?@I\???(?@a????&?i??T>k????Unknown
X7HostEqual"Equal(1?l????@9?l????@A?l????@I?l????@a??P[Q#?i???N?????Unknown
~8HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1H?z?G@9H?z?G@AH?z?G@IH?z?G@a?!Qv:?"?i??1B?????Unknown
?9HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1?E????B@9?E????B@A??v??@I??v??@a??s6'??ib?ks?????Unknown
a:HostIdentity"Identity(1?Zd;???9?Zd;???A?Zd;???I?Zd;???aI??,J??>i     ???Unknown?*?6
?HostMaxPoolGrad":gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGrad(1?Q??(?@9?Q??(?@A?Q??(?@I?Q??(?@aJ???????iJ????????Unknown
pHost_FusedConv2D"sequential/conv2d/Relu(1?v??Z??@9?v??Z??@A?v??Z??@I?v??Z??@a?<?*??i??{;????Unknown
?HostConv2DBackpropFilter";gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilter(1??C?l|?@9??C?l|?@A??C?l|?@I??C?l|?@aFU&;խ??i??l&????Unknown
^HostGatherV2"GatherV2(1?&1?M?@9?&1?M?@A?&1?M?@I?&1?M?@a7?b$?3??iY?BQ?????Unknown
~HostReluGrad"(gradient_tape/sequential/conv2d/ReluGrad(1??"????@9??"????@A??"????@I??"????@a?s?Z亰?iӅ???????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1B`??"*?@9B`??"*?@AB`??"*?@IB`??"*?@a?Jz????i?'8??q???Unknown
{HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1h??|?&?@9h??|?&?@Ah??|?&?@Ih??|?&?@a{$?d@??i???????Unknown
?HostBiasAddGrad"3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGrad(1w??/??@9w??/??@Aw??/??@Iw??/??@a?+d⨩?i????????Unknown
u	HostMaxPool" sequential/max_pooling2d/MaxPool(1????K?@9????K?@A????K?@I????K?@a?b?????in??????Unknown
y
HostMatMul"%gradient_tape/sequential/dense/MatMul(1J+?.w@9J+?.w@AJ+?.w@IJ+?.w@a??10Ɯ?i??G?????Unknown
qHostCast"sequential/dropout/dropout/Cast(1??~j?li@9??~j?li@A??~j?li@I??~j?li@aFe?I뎏?iR?nAXT???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1??"??Nd@9??"??Nd@A??"??Nd@I??"??Nd@a??05??i.$m-????Unknown
oHostMul"sequential/dropout/dropout/Mul(1bX9??c@9bX9??c@AbX9??c@IbX9??c@aa?:??i?11???Unknown
HostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1??C?lkb@9??C?lkb@A??C?lkb@I??C?lkb@ap????܆?iPo?u???Unknown
}HostMul",gradient_tape/sequential/dropout/dropout/Mul(1??? ??a@9??? ??a@A??? ??a@I??? ??a@a?&S7V??i?Y??????Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1???K7?`@9???K7?`@A???K7?`@I???K7?`@aD?\????i????!???Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1?? ?r?_@9?? ?r?_@A?? ?r?_@I?? ?r?_@a{1t??ib?ԝ	p???Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(1?S㥛T]@9?S㥛T]@A?S㥛T]@I?S㥛T]@a???U 4??iQ+?ٸ???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1F????XJ@9F????XJ@AF????XJ@IF????XJ@a?I?X?Yp?i??ܗ?????Unknown?
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1u?V>G@9u?V>G@Au?V>G@Iu?V>G@a???0??l?i=?gg????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1?ZdKD@9?ZdKD@A??v???@@I??v???@@aj??Z?d?i???f???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1T㥛?0@@9T㥛?0@@AT㥛?0@@IT㥛?0@@aOOfj?d?iT k???Unknown
iHostWriteSummary"WriteSummary(1??C?l?:@9??C?l?:@A??C?l?:@I??C?l?:@a????v`?in??Q?/???Unknown?
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1h??|??8@9h??|??8@Ah??|??8@Ih??|??8@a??}???^?iL{W?_????Unknown
mHostSoftmax"sequential/dense/Softmax(1o??ʡ7@9o??ʡ7@Ao??ʡ7@Io??ʡ7@a???"BU]?i??hd
N???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1???SA@9???SA@A??(\?b6@I??(\?b6@a?k???[?i?"5??[???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(133333?0@933333?0@A33333?0@I33333?0@a?6?}G?T?i?	`f???Unknown
gHostStridedSlice"strided_slice(1?/?$?-@9?/?$?-@A?/?$?-@I?/?$?-@a??RR?i?<i?o???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1X9??v~,@9X9??v~,@AX9??v~,@IX9??v~,@a?"X?Q?i???`x???Unknown
ZHostArgMax"ArgMax(1?V-)@9?V-)@A?V-)@I?V-)@a?.t??O?i???0????Unknown
?HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1+????'@9+????'@A+????'@I+????'@a???i?M?iJ%????Unknown
d HostDataset"Iterator::Model(1V-???9@9V-???9@AT㥛?`'@IT㥛?`'@a?>X<?M?iZ;??????Unknown
?!HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1?x?&1H'@9?x?&1H'@A?x?&1H'@I?x?&1H'@a ܆?L?ip?????Unknown
V"HostSum"Sum_2(1?????&@9?????&@A?????&@I?????&@a^]??^K?iW??9?????Unknown
v#HostAssignAddVariableOp"AssignAddVariableOp_2(133333?%@933333?%@A33333?%@I33333?%@a?	7?Y?J?i???????Unknown
Y$HostPow"Adam/Pow(1???Mb%@9???Mb%@A???Mb%@I???Mb%@a??rA%J?i5Z`8????Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_3(19??v?_#@99??v?_#@A9??v?_#@I9??v?_#@a?{??>H?i/? p;????Unknown
x&HostDataset"#Iterator::Model::ParallelMapV2::Zip(1B`??"?U@9B`??"?U@Ad;?O??"@Id;?O??"@aCϲ2cG?i?Š<????Unknown
['HostAddV2"Adam/add(1?K7?A?!@9?K7?A?!@A?K7?A?!@I?K7?A?!@a???3>0F?i??-L?????Unknown
l(HostIteratorGetNext"IteratorGetNext(1Zd;ߏ!@9Zd;ߏ!@AZd;ߏ!@IZd;ߏ!@a>X ?w?E?i?Kj????Unknown
t)HostAssignAddVariableOp"AssignAddVariableOp(1?Q??k!@9?Q??k!@A?Q??k!@I?Q??k!@a?KfX?E?iD?)@{????Unknown
e*Host
LogicalAnd"
LogicalAnd(1????K7 @9????K7 @A????K7 @I????K7 @a{?}?? D?i?0q?????Unknown?
?+HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1B`??"[@9B`??"[@AB`??"[@IB`??"[@a?*??uC?i????`????Unknown
{,HostSum"*categorical_crossentropy/weighted_loss/Sum(1?z?G?@9?z?G?@A?z?G?@I?z?G?@a??62/*C?i9??p+????Unknown
b-HostDivNoNan"div_no_nan_1(1!?rh?m@9!?rh?m@A!?rh?m@I!?rh?m@a?윤A?i????????Unknown
?.HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1?Zd;@9?Zd;@A?Zd;@I?Zd;@a?k????@?i?T>?????Unknown
?/HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1J+?@9J+?@AJ+?@IJ+?@a?a??@?i?X?+????Unknown
?0HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1?/?$@9?/?$@A?/?$@I?/?$@a?????@?ij?Ǝ3????Unknown
\1HostArgMax"ArgMax_1(1?$???@9?$???@A?$???@I?$???@a???????i????2????Unknown
`2HostGatherV2"
GatherV2_1(1Zd;?O?@9Zd;?O?@AZd;?O?@IZd;?O?@a4??Z???i??)????Unknown
t3HostReadVariableOp"Adam/Cast/ReadVariableOp(1?n??J@9?n??J@A?n??J@I?n??J@anL??&>?i@;h??????Unknown
v4HostAssignAddVariableOp"AssignAddVariableOp_4(1?5^?I?@9?5^?I?@A?5^?I?@I?5^?I?@akf?J?9?ih??????Unknown
[5HostPow"
Adam/Pow_1(1\???(?@9\???(?@A\???(?@I\???(?@a?~4ʬ?8?i??]?3????Unknown
X6HostEqual"Equal(1?l????@9?l????@A?l????@I?l????@a?m\?5?i+:S??????Unknown
~7HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1H?z?G@9H?z?G@AH?z?G@IH?z?G@a??^?54?i??U[????Unknown
?8HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1?E????B@9?E????B@A??v??@I??v??@a\??A?0?iv?ʝu????Unknown
a9HostIdentity"Identity(1?Zd;???9?Zd;???A?Zd;???I?Zd;???ax	?FL?i?????????Unknown?2CPU