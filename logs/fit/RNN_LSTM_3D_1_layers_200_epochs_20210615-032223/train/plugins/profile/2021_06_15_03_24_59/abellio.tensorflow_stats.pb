"?2
BHostIDLE"IDLE1%?????@A%?????@a?ٶ?I^??i?ٶ?I^???Unknown
?HostMaxPoolGrad":gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGrad(1??(\Ϥ?@9??(\Ϥ?@A??(\Ϥ?@I??(\Ϥ?@a?!?ٍR??i?P?vY???Unknown
pHost_FusedConv2D"sequential/conv2d/Relu(1?G?z?,?@9?G?z?,?@A?G?z?,?@I?G?z?,?@ay????n??iʫVVR'???Unknown
?HostConv2DBackpropFilter";gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilter(11??X?@91??X?@A1??X?@I1??X?@a????*??izK??L???Unknown
^HostGatherV2"GatherV2(1m??????@9m??????@Am??????@Im??????@a?7??)???i??X????Unknown
~HostReluGrad"(gradient_tape/sequential/conv2d/ReluGrad(1????Ë@9????Ë@A????Ë@I????Ë@a³	>??iK???	???Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1??ʡƉ@9??ʡƉ@A??ʡƉ@I??ʡƉ@a-<+(??i??J?????Unknown
?HostBiasAddGrad"3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGrad(1X9??v%?@9X9??v%?@AX9??v%?@IX9??v%?@a?????i?W??????Unknown
{	HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1??? ?]?@9??? ?]?@A??? ?]?@I??? ?]?@ak? ??*??io0?ޤ???Unknown
u
HostMaxPool" sequential/max_pooling2d/MaxPool(1??ʡE?@9??ʡE?@A??ʡE?@I??ʡE?@a?2??????i?>??D???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1#??~j?z@9#??~j?z@A#??~j?z@I#??~j?z@a???+??ipC97C????Unknown
qHostCast"sequential/dropout/dropout/Cast(1?&1?~q@9?&1?~q@A?&1?~q@I?&1?~q@a9???w??ii?U?"???Unknown
oHostMul"sequential/dropout/dropout/Mul(1q=
ף?n@9q=
ף?n@Aq=
ף?n@Iq=
ף?n@a?[[???iq??3[???Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1???S?um@9???S?um@A???S?um@I???S?um@a	G?j?;??i?Xo?????Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1?/?$vf@9?/?$vf@A?/?$vf@I?/?$vf@ah\ƲwGz?iF?Ԡ?????Unknown
}HostMul",gradient_tape/sequential/dropout/dropout/Mul(1?G?zdd@9?G?zdd@A?G?zdd@I?G?zdd@a7?y??w?i???/N???Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(1X9???b@9X9???b@AX9???b@IX9???b@a<?? }
v?im*c0???Unknown
HostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1#??~j|`@9#??~j|`@A#??~j|`@I#??~j|`@a??"@?Is?iŲ???V???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1P??n?\@9P??n?\@AP??n?\@IP??n?\@alR???p?i?W?ѓx???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1??K7??I@9??K7??I@A??K7??I@I??K7??I@a:Hq?4^?iA?vH?????Unknown?
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1?&1?E@9?&1?E@A?&1?E@I?&1?E@a?T?֎?X?ik????????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1??Q??E@9??Q??E@AL7?A`5B@IL7?A`5B@af?A??MU?iYx?b?????Unknown
iHostWriteSummary"WriteSummary(1??v???9@9??v???9@A??v???9@I??v???9@a??L??N?i?KxF5????Unknown?
dHostDataset"Iterator::Model(1P??n?B@9P??n?B@AD?l???6@ID?l???6@a+?g??^J?iy%??̬???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1NbX9?4@9NbX9?4@ANbX9?4@INbX9?4@a?Nd??H?i'9??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1bX9??4@9bX9??4@AbX9??4@IbX9??4@aG?!Q?+H?i??e??????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1??/??3@9??/??3@A??/??3@I??/??3@aycyFG?iQ?sʾ???Unknown
mHostSoftmax"sequential/dense/Softmax(1?S㥛?3@9?S㥛?3@A?S㥛?3@I?S㥛?3@ak? G?ilެ??????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(11?ZD;@91?ZD;@A?????0@I?????0@a?y?
,?C?i
???~????Unknown
`HostGatherV2"
GatherV2_1(1;?O??..@9;?O??..@A;?O??..@I;?O??..@a=????A?i?Ǧ??????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1???Q?,@9???Q?,@A???Q?,@I???Q?,@aE]R?j?@?ip\aC%????Unknown
g HostStridedSlice"strided_slice(1-????+@9-????+@A-????+@I-????+@a?,??S@?i?R[:????Unknown
[!HostAddV2"Adam/add(1?t?)@9?t?)@A?t?)@I?t?)@a??HʈY=?iڛt9?????Unknown
l"HostIteratorGetNext"IteratorGetNext(1?? ?r?&@9?? ?r?&@A?? ?r?&@I?? ?r?&@a.??3?:?i????>????Unknown
?#HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1T㥛??$@9T㥛??$@AT㥛??$@IT㥛??$@a,??1m8?i?G??L????Unknown
?$HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1??v???$@9??v???$@A??v???$@I??v???$@a??? f8?i?\FY????Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1??? ?2$@9??? ?2$@A??? ?2$@I??? ?2$@a< _8??7?i?h?vM????Unknown
x&HostDataset"#Iterator::Model::ParallelMapV2::Zip(1????x?T@9????x?T@A{?G?:"@I{?G?:"@a?ʅ'T5?itY???????Unknown
Z'HostArgMax"ArgMax(1? ?rh@9? ?rh@A? ?rh@I? ?rh@a9??Ϛ,2?i?V??=????Unknown
?(HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1?ʡE?s@9?ʡE?s@A?ʡE?s@I?ʡE?s@a????Z?1?i4Z?w????Unknown
{)HostSum"*categorical_crossentropy/weighted_loss/Sum(17?A`?P@97?A`?P@A7?A`?P@I7?A`?P@a??3?;&1?iwZ?_?????Unknown
V*HostSum"Sum_2(1??S??@9??S??@A??S??@I??S??@a??z1?i-?~B?????Unknown
e+Host
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@a>???0?i??Q??????Unknown?
?,HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1?n??J@9?n??J@A?n??J@I?n??J@a???%??0?ie=???????Unknown
?-HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1?????@9?????@A?????@I?????@aK?ѓ?+?i~z???????Unknown
\.HostArgMax"ArgMax_1(1F?????@9F?????@AF?????@IF?????@a???2^?*?i8???@????Unknown
[/HostPow"
Adam/Pow_1(1?z?G?@9?z?G?@A?z?G?@I?z?G?@a??a??*?iV??>?????Unknown
t0HostAssignAddVariableOp"AssignAddVariableOp(1     ?@9     ?@A     ?@I     ?@aܩA?qy$?iq???4????Unknown
X1HostEqual"Equal(1}?5^?I@9}?5^?I@A}?5^?I@I}?5^?I@aQ?6
"!?i3^??F????Unknown
?2HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1^?I?>@9^?I?>@Ah??|?5@Ih??|?5@aCY8!?i??sWX????Unknown
Y3HostPow"Adam/Pow(1o??ʡ@9o??ʡ@Ao??ʡ@Io??ʡ@a??J???i?Ї5????Unknown
?4HostReadVariableOp"(sequential/conv2d/BiasAdd/ReadVariableOp(1-??????9-??????A-??????I-??????a????d??i}R???????Unknown
a5HostIdentity"Identity(1????K7??9????K7??A????K7??I????K7??a??_?A?i?????????Unknown?*?1
?HostMaxPoolGrad":gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGrad(1??(\Ϥ?@9??(\Ϥ?@A??(\Ϥ?@I??(\Ϥ?@a8r?G??i8r?G???Unknown
pHost_FusedConv2D"sequential/conv2d/Relu(1?G?z?,?@9?G?z?,?@A?G?z?,?@I?G?z?,?@arh?u>X??iUm??$????Unknown
?HostConv2DBackpropFilter";gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilter(11??X?@91??X?@A1??X?@I1??X?@a??>ũU??iT?p??????Unknown
^HostGatherV2"GatherV2(1m??????@9m??????@Am??????@Im??????@aN????:??it?L?ո???Unknown
~HostReluGrad"(gradient_tape/sequential/conv2d/ReluGrad(1????Ë@9????Ë@A????Ë@I????Ë@a?)??????i?[o_????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1??ʡƉ@9??ʡƉ@A??ʡƉ@I??ʡƉ@a?Z??????i??ZCyr???Unknown
?HostBiasAddGrad"3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGrad(1X9??v%?@9X9??v%?@AX9??v%?@IX9??v%?@ai???=ĩ?i&o&"????Unknown
{HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1??? ?]?@9??? ?]?@A??? ?]?@I??? ?]?@a?D??????iq???????Unknown
u	HostMaxPool" sequential/max_pooling2d/MaxPool(1??ʡE?@9??ʡE?@A??ʡE?@I??ʡE?@a??r?????i<??????Unknown
y
HostMatMul"%gradient_tape/sequential/dense/MatMul(1#??~j?z@9#??~j?z@A#??~j?z@I#??~j?z@a?JQ9????i?7?!K????Unknown
qHostCast"sequential/dropout/dropout/Cast(1?&1?~q@9?&1?~q@A?&1?~q@I?&1?~q@a??ZV?y??i?4|V???Unknown
oHostMul"sequential/dropout/dropout/Mul(1q=
ף?n@9q=
ף?n@Aq=
ף?n@Iq=
ף?n@a?*????i_??????Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1???S?um@9???S?um@A???S?um@I???S?um@a?2??e??i???t6b???Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1?/?$vf@9?/?$vf@A?/?$vf@I?/?$vf@aq?BA??i???:????Unknown
}HostMul",gradient_tape/sequential/dropout/dropout/Mul(1?G?zdd@9?G?zdd@A?G?zdd@I?G?zdd@a?I%?s???i-?P?!???Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(1X9???b@9X9???b@AX9???b@IX9???b@a??	T????i?????t???Unknown
HostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1#??~j|`@9#??~j|`@A#??~j|`@I#??~j|`@a??ۓ-Z??i?7T????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1P??n?\@9P??n?\@AP??n?\@IP??n?\@a%k???i?"ƖK????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1??K7??I@9??K7??I@A??K7??I@I??K7??I@a?????l?i??qW	???Unknown?
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1?&1?E@9?&1?E@A?&1?E@I?&1?E@a?-w?ng?i+ux2???Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1??Q??E@9??Q??E@AL7?A`5B@IL7?A`5B@a?ccLEd?it???F???Unknown
iHostWriteSummary"WriteSummary(1??v???9@9??v???9@A??v???9@I??v???9@a?"3???\?i(
!U???Unknown?
dHostDataset"Iterator::Model(1P??n?B@9P??n?B@AD?l???6@ID?l???6@a}?cA)Y?i?٪??a???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1NbX9?4@9NbX9?4@ANbX9?4@INbX9?4@a?!i?wSW?im??TVm???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1bX9??4@9bX9??4@AbX9??4@IbX9??4@a%f?V?i?N??x???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1??/??3@9??/??3@A??/??3@I??/??3@a=???c%V?i????????Unknown
mHostSoftmax"sequential/dense/Softmax(1?S㥛?3@9?S㥛?3@A?S㥛?3@I?S㥛?3@ao<#a{V?i@00w?????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(11?ZD;@91?ZD;@A?????0@I?????0@a??v"??R?i?kAIG????Unknown
`HostGatherV2"
GatherV2_1(1;?O??..@9;?O??..@A;?O??..@I;?O??..@a?ˌ???P?i2
??????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1???Q?,@9???Q?,@A???Q?,@I???Q?,@aRP???P?i9????????Unknown
gHostStridedSlice"strided_slice(1-????+@9-????+@A-????+@I-????+@a"7???O?iG????????Unknown
[ HostAddV2"Adam/add(1?t?)@9?t?)@A?t?)@I?t?)@an??>?K?i<K?|????Unknown
l!HostIteratorGetNext"IteratorGetNext(1?? ?r?&@9?? ?r?&@A?? ?r?&@I?? ?r?&@aU??Q?I?i?w?ݽ???Unknown
?"HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1T㥛??$@9T㥛??$@AT㥛??$@IT㥛??$@aHc?A?=G?i*􁃬????Unknown
?#HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1??v???$@9??v???$@A??v???$@I??v???$@a??_W?6G?i̗Az????Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_2(1??? ?2$@9??? ?2$@A??? ?2$@I??? ?2$@aѠ~?|F?i?kC????Unknown
x%HostDataset"#Iterator::Model::ParallelMapV2::Zip(1????x?T@9????x?T@A{?G?:"@I{?G?:"@a?(??.KD?i?b?,????Unknown
Z&HostArgMax"ArgMax(1? ?rh@9? ?rh@A? ?rh@I? ?rh@aN????JA?i????~????Unknown
?'HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1?ʡE?s@9?ʡE?s@A?ʡE?s@I?ʡE?s@a???x?@?i?ʌ??????Unknown
{(HostSum"*categorical_crossentropy/weighted_loss/Sum(17?A`?P@97?A`?P@A7?A`?P@I7?A`?P@a>4?<Q@?i?????????Unknown
V)HostSum"Sum_2(1??S??@9??S??@A??S??@I??S??@aV?ј3@?ix????????Unknown
e*Host
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@a??^?m???iJ|p?????Unknown?
?+HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1?n??J@9?n??J@A?n??J@I?n??J@a?U?z?~??i5mKE?????Unknown
?,HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1?????@9?????@A?????@I?????@a?(o??9?i?h&?????Unknown
\-HostArgMax"ArgMax_1(1F?????@9F?????@AF?????@IF?????@a-F???9?iC}?x)????Unknown
[.HostPow"
Adam/Pow_1(1?z?G?@9?z?G?@A?z?G?@I?z?G?@a|ˍ?Vx9?i???X????Unknown
t/HostAssignAddVariableOp"AssignAddVariableOp(1     ?@9     ?@A     ?@I     ?@aٹc?%{3?is;???????Unknown
X0HostEqual"Equal(1}?5^?I@9}?5^?I@A}?5^?I@I}?5^?I@a}???>M0?iR???????Unknown
?1HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1^?I?>@9^?I?>@Ah??|?5@Ih??|?5@a????A0?i5???????Unknown
Y2HostPow"Adam/Pow(1o??ʡ@9o??ʡ@Ao??ʡ@Io??ʡ@aG]???N*?iZ??~????Unknown
?3HostReadVariableOp"(sequential/conv2d/BiasAdd/ReadVariableOp(1-??????9-??????A-??????I-??????abQb??i-??}????Unknown
a4HostIdentity"Identity(1????K7??9????K7??A????K7??I????K7??aԇ?`?B?i     ???Unknown?2CPU