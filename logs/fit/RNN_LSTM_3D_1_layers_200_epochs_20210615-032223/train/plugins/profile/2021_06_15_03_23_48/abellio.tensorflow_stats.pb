"?8
BHostIDLE"IDLE1ffff??@Affff??@aV<HY??iV<HY???Unknown
?HostMaxPoolGrad":gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGrad(1???̬??@9???̬??@A???̬??@I???̬??@aI?????i?)???????Unknown
pHost_FusedConv2D"sequential/conv2d/Relu(1y?&1?M?@9y?&1?M?@Ay?&1?M?@Iy?&1?M?@aG??Qg??i(??/է???Unknown
?HostConv2DBackpropFilter";gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilter(1?"??>??@9?"??>??@A?"??>??@I?"??>??@a:)l߯?i?=ǥ???Unknown
^HostGatherV2"GatherV2(1?????'?@9?????'?@A?????'?@I?????'?@a6?_?^??i;????????Unknown
?HostBiasAddGrad"3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGrad(1? ?r?r?@9? ?r?r?@A? ?r?r?@I? ?r?r?@a?0|?I??iI?+@????Unknown
~HostReluGrad"(gradient_tape/sequential/conv2d/ReluGrad(1H?z.?@9H?z.?@AH?z.?@IH?z.?@auK?d???i?/Q?q????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1?C?lg??@9?C?lg??@A?C?lg??@I?C?lg??@a???Z???i?&?^????Unknown
u	HostMaxPool" sequential/max_pooling2d/MaxPool(1??Mb?@9??Mb?@A??Mb?@I??Mb?@a%????ő?i5E%??????Unknown
{
HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1??ʡ|?@9??ʡ|?@A??ʡ|?@I??ʡ|?@a?*&D?P??i?vF????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(19??v??z@99??v??z@A9??v??z@I9??v??z@aV*????i??b???Unknown
HostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1????Kw@9????Kw@A????Kw@I????Kw@a? ??7́?ig2??c???Unknown
qHostCast"sequential/dropout/dropout/Cast(1?&1??p@9?&1??p@A?&1??p@I?&1??p@aV???I?y?ivߑo????Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1???K7p@9???K7p@A???K7p@I???K7p@a(?G??x?i?4m?????Unknown
}HostMul",gradient_tape/sequential/dropout/dropout/Mul(1?Q??/n@9?Q??/n@A?Q??/n@I?Q??/n@a?W#>w?i????????Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(1??n??k@9??n??k@A??n??k@I??n??k@a`??nau?i??O"???Unknown
oHostMul"sequential/dropout/dropout/Mul(1?/?$?i@9?/?$?i@A?/?$?i@I?/?$?i@a?0???s?i'izJ???Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1V-???b@9V-???b@AV-???b@IV-???b@a??Ӱ|?l?i?<+??f???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1V-??b@9V-??b@AV-??b@IV-??b@a?f5???k?iBr2?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1?rh??N@9?rh??N@A?rh??N@I?rh??N@aǷ%]?/W?iI????Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1X9??N@9X9??N@AX9??N@IX9??N@a??So?W?i? j?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1V-?]J@9V-?]J@AV-?]J@IV-?]J@a?0?&MT?iG]?ң???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1?$??sH@9?$??sH@Au?V?D@Iu?V?D@a????P?i?JO??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1R???qB@9R???qB@AR???qB@IR???qB@a??9??gL?iSYz??????Unknown
iHostWriteSummary"WriteSummary(1j?t???@9j?t???@Aj?t???@Ij?t???@ae???\H?ig	????Unknown?
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1+??9@9+??9@A+??9@I+??9@a{??5ĵC?i|tz ????Unknown
mHostSoftmax"sequential/dense/Softmax(1bX9?h7@9bX9?h7@AbX9?h7@IbX9?h7@a`??GB?i??d?????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1sh??|??@9sh??|??@A?????l6@I?????l6@a????QDA?ie?? ?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1????ƫ5@9????ƫ5@A????ƫ5@I????ƫ5@a???z??@?iOT%?????Unknown
gHostStridedSlice"strided_slice(1?Zd;?3@9?Zd;?3@A?Zd;?3@I?Zd;?3@aRZţ=?iZ5ȅ?????Unknown
[HostAddV2"Adam/add(1????2@9????2@A????2@I????2@a??x???;?is?c.????Unknown
? HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(11?Z$1@91?Z$1@A1?Z$1@I1?Z$1@aSƿ??e:?il?: {????Unknown
Z!HostArgMax"ArgMax(1????x?.@9????x?.@A????x?.@I????x?.@a?4???7?i????n????Unknown
d"HostDataset"Iterator::Model(1?x?&1hB@9?x?&1hB@A???K7I.@I???K7I.@a0j?8?Q7?i,???X????Unknown
Y#HostPow"Adam/Pow(1?$???,@9?$???,@A?$???,@I?$???,@a?9??%6?i???????Unknown
?$HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1??x?&?*@9??x?&?*@A??x?&?*@I??x?&?*@aM
5i?4?i]??A?????Unknown
v%HostAssignAddVariableOp"AssignAddVariableOp_2(1-????F)@9-????F)@A-????F)@I-????F)@a?8q?Kv3?i??H????Unknown
x&HostDataset"#Iterator::Model::ParallelMapV2::Zip(1??ʡ?W@9??ʡ?W@A}?5^??&@I}?5^??&@a]?.ߋ1?i\?+?O????Unknown
{'HostSum"*categorical_crossentropy/weighted_loss/Sum(1??C?lg&@9??C?lg&@A??C?lg&@I??C?lg&@a#?d?-@1?i????w????Unknown
e(Host
LogicalAnd"
LogicalAnd(1??C?lg%@9??C?lg%@A??C?lg%@I??C?lg%@a);{0?i!???????Unknown?
l)HostIteratorGetNext"IteratorGetNext(1m?????$@9m?????$@Am?????$@Im?????$@a???1??/?i?<?݁????Unknown
t*HostAssignAddVariableOp"AssignAddVariableOp(1?? ?r?#@9?? ?r?#@A?? ?r?#@I?? ?r?#@aN??U[?.?i9?Fcl????Unknown
?+HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1L7?A`?"@9L7?A`?"@AL7?A`?"@IL7?A`?"@a?5=ֶ,?i?f??7????Unknown
?,HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1?G?zT"@9?G?zT"@A?G?zT"@I?G?zT"@a\?рB:,?i?s?t?????Unknown
[-HostPow"
Adam/Pow_1(1?I+"@9?I+"@A?I+"@I?I+"@a??V?3?+?iY???????Unknown
t.HostReadVariableOp"Adam/Cast/ReadVariableOp(1ˡE???!@9ˡE???!@AˡE???!@IˡE???!@aQ? u?+?ih?4?r????Unknown
V/HostSum"Sum_2(1????Ƌ @9????Ƌ @A????Ƌ @I????Ƌ @a(:?}?z)?iܱl?
????Unknown
`0HostGatherV2"
GatherV2_1(1??C??@9??C??@A??C??@I??C??@a?NkU??(?i????????Unknown
\1HostArgMax"ArgMax_1(1?I+@9?I+@A?I+@I?I+@a??? ?'?i???????Unknown
?2HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1D?l??)@9D?l??)@AD?l??)@ID?l??)@a??;???%?i?%??`????Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_4(1?l????@9?l????@A?l????@I?l????@a
?A?c  ?i?i?b????Unknown
~4HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1fffff?@9fffff?@Afffff?@Ifffff?@a?z????i???g?????Unknown
X5HostEqual"Equal(11?Z?@91?Z?@A1?Z?@I1?Z?@a??"???i???????Unknown
?6HostReadVariableOp"(sequential/conv2d/BiasAdd/ReadVariableOp(1??ʡE@9??ʡE@A??ʡE@I??ʡE@a??'???i??k$?????Unknown
b7HostDivNoNan"div_no_nan_1(1h??|?5@9h??|?5@Ah??|?5@Ih??|?5@a#?qe??i?g?w????Unknown
?8HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1h??|?uA@9h??|?uA@A??MbX	@I??MbX	@a?????iJ???????Unknown
?9HostReadVariableOp"'sequential/conv2d/Conv2D/ReadVariableOp(1/?$??@9/?$??@A/?$??@I/?$??@aK??6?(?iQ?}?????Unknown
w:HostReadVariableOp"div_no_nan_1/ReadVariableOp(1??C?l??9??C?l??A??C?l??I??C?l??a?Y~v ?iJt?y?????Unknown
a;HostIdentity"Identity(1?G?z??9?G?z??A?G?z??I?G?z??a?'?E=??>i?????????Unknown?*?7
?HostMaxPoolGrad":gradient_tape/sequential/max_pooling2d/MaxPool/MaxPoolGrad(1???̬??@9???̬??@A???̬??@I???̬??@a???
s??i???
s???Unknown
pHost_FusedConv2D"sequential/conv2d/Relu(1y?&1?M?@9y?&1?M?@Ay?&1?M?@Iy?&1?M?@a??&?????i?M{?v.???Unknown
?HostConv2DBackpropFilter";gradient_tape/sequential/conv2d/Conv2D/Conv2DBackpropFilter(1?"??>??@9?"??>??@A?"??>??@I?"??>??@aI???g??i^??b????Unknown
^HostGatherV2"GatherV2(1?????'?@9?????'?@A?????'?@I?????'?@a?כ?'??i?????????Unknown
?HostBiasAddGrad"3gradient_tape/sequential/conv2d/BiasAdd/BiasAddGrad(1? ?r?r?@9? ?r?r?@A? ?r?r?@I? ?r?r?@aIE??`???iOC
??????Unknown
~HostReluGrad"(gradient_tape/sequential/conv2d/ReluGrad(1H?z.?@9H?z.?@AH?z.?@IH?z.?@aMb?d??i?DK%?????Unknown
?HostRandomUniform"7sequential/dropout/dropout/random_uniform/RandomUniform(1?C?lg??@9?C?lg??@A?C?lg??@I?C?lg??@a??' ???i?MW?????Unknown
uHostMaxPool" sequential/max_pooling2d/MaxPool(1??Mb?@9??Mb?@A??Mb?@I??Mb?@aձA??h??i%?A i????Unknown
{	HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1??ʡ|?@9??ʡ|?@A??ʡ|?@I??ʡ|?@aH(?͍???i????+???Unknown
y
HostMatMul"%gradient_tape/sequential/dense/MatMul(19??v??z@99??v??z@A9??v??z@I9??v??z@a?T2??x??iNQù????Unknown
HostMul".gradient_tape/sequential/dropout/dropout/Mul_1(1????Kw@9????Kw@A????Kw@I????Kw@a?ӗ?o??iw???6{???Unknown
qHostCast"sequential/dropout/dropout/Cast(1?&1??p@9?&1??p@A?&1??p@I?&1??p@a?hIdQ??iˍ??|????Unknown
?HostGreaterEqual"'sequential/dropout/dropout/GreaterEqual(1???K7p@9???K7p@A???K7p@I???K7p@a>?H-???ip????X???Unknown
}HostMul",gradient_tape/sequential/dropout/dropout/Mul(1?Q??/n@9?Q??/n@A?Q??/n@I?Q??/n@a4??M?a??i???i????Unknown
rHost_FusedMatMul"sequential/dense/BiasAdd(1??n??k@9??n??k@A??n??k@I??n??k@a'!!?MY??iz???????Unknown
oHostMul"sequential/dropout/dropout/Mul(1?/?$?i@9?/?$?i@A?/?$?i@I?/?$?i@a?yf????ia=??r???Unknown
qHostMul" sequential/dropout/dropout/Mul_1(1V-???b@9V-???b@AV-???b@IV-???b@a?????=?i??c????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1V-??b@9V-??b@AV-??b@IV-??b@aeDW7?s~?in?8?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1?rh??N@9?rh??N@A?rh??N@I?rh??N@a?),Ri?i?aHB???Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(1X9??N@9X9??N@AX9??N@IX9??N@a?[]h??i?im9?Ձ ???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1V-?]J@9V-?]J@AV-?]J@IV-?]J@a???X?+f?i!#l?6???Unknown?
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1?$??sH@9?$??sH@Au?V?D@Iu?V?D@a?`???a?i??GH???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1R???qB@9R???qB@AR???qB@IR???qB@a?1 _?i
'??W???Unknown
iHostWriteSummary"WriteSummary(1j?t???@9j?t???@Aj?t???@Ij?t???@a?,???Z?i?e???Unknown?
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1+??9@9+??9@A+??9@I+??9@a0??1D?U?ib?>?o???Unknown
mHostSoftmax"sequential/dense/Softmax(1bX9?h7@9bX9?h7@AbX9?h7@IbX9?h7@a??&
?S?i??űy???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1sh??|??@9sh??|??@A?????l6@I?????l6@ap@5?=?R?i{+wd????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1????ƫ5@9????ƫ5@A????ƫ5@I????ƫ5@al?? ?8R?i?x?;????Unknown
gHostStridedSlice"strided_slice(1?Zd;?3@9?Zd;?3@A?Zd;?3@I?Zd;?3@a8J??#/P?i B?mS????Unknown
[HostAddV2"Adam/add(1????2@9????2@A????2@I????2@a?;$??fN?i?-?????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(11?Z$1@91?Z$1@A1?Z$1@I1?Z$1@a?; T?L?i?/"????Unknown
Z HostArgMax"ArgMax(1????x?.@9????x?.@A????x?.@I????x?.@a"??Y?I?i&?$F?????Unknown
d!HostDataset"Iterator::Model(1?x?&1hB@9?x?&1hB@A???K7I.@I???K7I.@a?1??iwI?irr? ?????Unknown
Y"HostPow"Adam/Pow(1?$???,@9?$???,@A?$???,@I?$???,@a????/H?iZo??????Unknown
?#HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1??x?&?*@9??x?&?*@A??x?&?*@I??x?&?*@a?;8?qF?i?v???????Unknown
v$HostAssignAddVariableOp"AssignAddVariableOp_2(1-????F)@9-????F)@A-????F)@I-????F)@aBO???@E?i?????????Unknown
x%HostDataset"#Iterator::Model::ParallelMapV2::Zip(1??ʡ?W@9??ʡ?W@A}?5^??&@I}?5^??&@a?졜a)C?i8???????Unknown
{&HostSum"*categorical_crossentropy/weighted_loss/Sum(1??C?lg&@9??C?lg&@A??C?lg&@I??C?lg&@a?b$???B?iQ??j????Unknown
e'Host
LogicalAnd"
LogicalAnd(1??C?lg%@9??C?lg%@A??C?lg%@I??C?lg%@a?v?u?A?i?;r??????Unknown?
l(HostIteratorGetNext"IteratorGetNext(1m?????$@9m?????$@Am?????$@Im?????$@a?B??LA?i6L$?=????Unknown
t)HostAssignAddVariableOp"AssignAddVariableOp(1?? ?r?#@9?? ?r?#@A?? ?r?#@I?? ?r?#@a?bt?m?@?iO??2m????Unknown
?*HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1L7?A`?"@9L7?A`?"@AL7?A`?"@IL7?A`?"@aD?w?[??iCة?X????Unknown
?+HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1?G?zT"@9?G?zT"@A?G?zT"@I?G?zT"@a? ?|}?>?icoY3????Unknown
[,HostPow"
Adam/Pow_1(1?I+"@9?I+"@A?I+"@I?I+"@a????xQ>?iԆuB?????Unknown
t-HostReadVariableOp"Adam/Cast/ReadVariableOp(1ˡE???!@9ˡE???!@AˡE???!@IˡE???!@a? -?A>?iT)???????Unknown
V.HostSum"Sum_2(1????Ƌ @9????Ƌ @A????Ƌ @I????Ƌ @a}?J?o?;?i?r???????Unknown
`/HostGatherV2"
GatherV2_1(1??C??@9??C??@A??C??@I??C??@aڱ???:?i;????????Unknown
\0HostArgMax"ArgMax_1(1?I+@9?I+@A?I+@I?I+@a2?p??9?iV????????Unknown
?1HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1D?l??)@9D?l??)@AD?l??)@ID?l??)@a?v???7?iE?	??????Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_4(1?l????@9?l????@A?l????@I?l????@a_\p?1?i(N?????Unknown
~3HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1fffff?@9fffff?@Afffff?@Ifffff?@aO?g~].?i?????????Unknown
X4HostEqual"Equal(11?Z?@91?Z?@A1?Z?@I1?Z?@a;?B??.?i?|?b?????Unknown
?5HostReadVariableOp"(sequential/conv2d/BiasAdd/ReadVariableOp(1??ʡE@9??ʡE@A??ʡE@I??ʡE@a̱3#?(?i???49????Unknown
b6HostDivNoNan"div_no_nan_1(1h??|?5@9h??|?5@Ah??|?5@Ih??|?5@a??Nj??&?i?RCB?????Unknown
?7HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1h??|?uA@9h??|?uA@A??MbX	@I??MbX	@aW?]?O%?i???;?????Unknown
?8HostReadVariableOp"'sequential/conv2d/Conv2D/ReadVariableOp(1/?$??@9/?$??@A/?$??@I/?$??@a?v??w ?i?ء?????Unknown
w9HostReadVariableOp"div_no_nan_1/ReadVariableOp(1??C?l??9??C?l??A??C?l??I??C?l??aj ????i?~?ԓ????Unknown
a:HostIdentity"Identity(1?G?z??9?G?z??A?G?z??I?G?z??a=?Q ?
?i     ???Unknown?2CPU