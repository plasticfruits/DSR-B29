èú
Ñ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878Ø

relu_dense_Qt_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_namerelu_dense_Qt_1/kernel

*relu_dense_Qt_1/kernel/Read/ReadVariableOpReadVariableOprelu_dense_Qt_1/kernel*
_output_shapes

:@*
dtype0

relu_dense_Qt_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namerelu_dense_Qt_1/bias
y
(relu_dense_Qt_1/bias/Read/ReadVariableOpReadVariableOprelu_dense_Qt_1/bias*
_output_shapes
:@*
dtype0

relu_dense_Qt_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_namerelu_dense_Qt_2/kernel

*relu_dense_Qt_2/kernel/Read/ReadVariableOpReadVariableOprelu_dense_Qt_2/kernel*
_output_shapes

:@@*
dtype0

relu_dense_Qt_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namerelu_dense_Qt_2/bias
y
(relu_dense_Qt_2/bias/Read/ReadVariableOpReadVariableOprelu_dense_Qt_2/bias*
_output_shapes
:@*
dtype0

relu_dense_Qt_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *'
shared_namerelu_dense_Qt_3/kernel

*relu_dense_Qt_3/kernel/Read/ReadVariableOpReadVariableOprelu_dense_Qt_3/kernel*
_output_shapes

:@ *
dtype0

relu_dense_Qt_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_namerelu_dense_Qt_3/bias
y
(relu_dense_Qt_3/bias/Read/ReadVariableOpReadVariableOprelu_dense_Qt_3/bias*
_output_shapes
: *
dtype0

Q_target_value/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameQ_target_value/kernel

)Q_target_value/kernel/Read/ReadVariableOpReadVariableOpQ_target_value/kernel*
_output_shapes

: *
dtype0
~
Q_target_value/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameQ_target_value/bias
w
'Q_target_value/bias/Read/ReadVariableOpReadVariableOpQ_target_value/bias*
_output_shapes
:*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¼
value²B¯ B¨
¤
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
loss
regularization_losses
	trainable_variables

	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
 bias
!regularization_losses
"trainable_variables
#	variables
$	keras_api
 
 
 
8
0
1
2
3
4
5
6
 7
8
0
1
2
3
4
5
6
 7
­
regularization_losses
%metrics
&non_trainable_variables
'layer_metrics

(layers
	trainable_variables

	variables
)layer_regularization_losses
 
b`
VARIABLE_VALUErelu_dense_Qt_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUErelu_dense_Qt_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
*metrics
+non_trainable_variables
,layer_metrics

-layers
trainable_variables
	variables
.layer_regularization_losses
b`
VARIABLE_VALUErelu_dense_Qt_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUErelu_dense_Qt_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
/metrics
0non_trainable_variables
1layer_metrics

2layers
trainable_variables
	variables
3layer_regularization_losses
b`
VARIABLE_VALUErelu_dense_Qt_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUErelu_dense_Qt_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
4metrics
5non_trainable_variables
6layer_metrics

7layers
trainable_variables
	variables
8layer_regularization_losses
a_
VARIABLE_VALUEQ_target_value/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEQ_target_value/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
­
!regularization_losses
9metrics
:non_trainable_variables
;layer_metrics

<layers
"trainable_variables
#	variables
=layer_regularization_losses
 
 
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
x
serving_default_statePlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
ø
StatefulPartitionedCallStatefulPartitionedCallserving_default_staterelu_dense_Qt_1/kernelrelu_dense_Qt_1/biasrelu_dense_Qt_2/kernelrelu_dense_Qt_2/biasrelu_dense_Qt_3/kernelrelu_dense_Qt_3/biasQ_target_value/kernelQ_target_value/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_35920819
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
û
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*relu_dense_Qt_1/kernel/Read/ReadVariableOp(relu_dense_Qt_1/bias/Read/ReadVariableOp*relu_dense_Qt_2/kernel/Read/ReadVariableOp(relu_dense_Qt_2/bias/Read/ReadVariableOp*relu_dense_Qt_3/kernel/Read/ReadVariableOp(relu_dense_Qt_3/bias/Read/ReadVariableOp)Q_target_value/kernel/Read/ReadVariableOp'Q_target_value/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_35921052
Ö
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamerelu_dense_Qt_1/kernelrelu_dense_Qt_1/biasrelu_dense_Qt_2/kernelrelu_dense_Qt_2/biasrelu_dense_Qt_3/kernelrelu_dense_Qt_3/biasQ_target_value/kernelQ_target_value/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_35921086º£
í

1__inference_Q_target_value_layer_call_fn_35921005

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Q_target_value_layer_call_and_return_conditional_losses_359206642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ö
Ô
&__inference_signature_wrapper_35920819	
state
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_359205682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate
²
µ
M__inference_relu_dense_Qt_2_layer_call_and_return_conditional_losses_35920610

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ï

2__inference_relu_dense_Qt_3_layer_call_fn_35920985

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_relu_dense_Qt_3_layer_call_and_return_conditional_losses_359206372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ï

2__inference_relu_dense_Qt_2_layer_call_fn_35920965

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_relu_dense_Qt_2_layer_call_and_return_conditional_losses_359206102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
±
´
L__inference_Q_target_value_layer_call_and_return_conditional_losses_35920996

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
²
µ
M__inference_relu_dense_Qt_3_layer_call_and_return_conditional_losses_35920637

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
©
Þ
/__inference_functional_3_layer_call_fn_35920904

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_functional_3_layer_call_and_return_conditional_losses_359207322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï

2__inference_relu_dense_Qt_1_layer_call_fn_35920945

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_relu_dense_Qt_1_layer_call_and_return_conditional_losses_359205832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
µ
M__inference_relu_dense_Qt_3_layer_call_and_return_conditional_losses_35920976

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ó&
Ü
$__inference__traced_restore_35921086
file_prefix+
'assignvariableop_relu_dense_qt_1_kernel+
'assignvariableop_1_relu_dense_qt_1_bias-
)assignvariableop_2_relu_dense_qt_2_kernel+
'assignvariableop_3_relu_dense_qt_2_bias-
)assignvariableop_4_relu_dense_qt_3_kernel+
'assignvariableop_5_relu_dense_qt_3_bias,
(assignvariableop_6_q_target_value_kernel*
&assignvariableop_7_q_target_value_bias

identity_9¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7ß
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*ë
valueáBÞ	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names 
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slicesØ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¦
AssignVariableOpAssignVariableOp'assignvariableop_relu_dense_qt_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¬
AssignVariableOp_1AssignVariableOp'assignvariableop_1_relu_dense_qt_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2®
AssignVariableOp_2AssignVariableOp)assignvariableop_2_relu_dense_qt_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¬
AssignVariableOp_3AssignVariableOp'assignvariableop_3_relu_dense_qt_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4®
AssignVariableOp_4AssignVariableOp)assignvariableop_4_relu_dense_qt_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¬
AssignVariableOp_5AssignVariableOp'assignvariableop_5_relu_dense_qt_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6­
AssignVariableOp_6AssignVariableOp(assignvariableop_6_q_target_value_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7«
AssignVariableOp_7AssignVariableOp&assignvariableop_7_q_target_value_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
²
µ
M__inference_relu_dense_Qt_2_layer_call_and_return_conditional_losses_35920956

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
­
þ
J__inference_functional_3_layer_call_and_return_conditional_losses_35920777

inputs
relu_dense_qt_1_35920756
relu_dense_qt_1_35920758
relu_dense_qt_2_35920761
relu_dense_qt_2_35920763
relu_dense_qt_3_35920766
relu_dense_qt_3_35920768
q_target_value_35920771
q_target_value_35920773
identity¢&Q_target_value/StatefulPartitionedCall¢'relu_dense_Qt_1/StatefulPartitionedCall¢'relu_dense_Qt_2/StatefulPartitionedCall¢'relu_dense_Qt_3/StatefulPartitionedCall½
'relu_dense_Qt_1/StatefulPartitionedCallStatefulPartitionedCallinputsrelu_dense_qt_1_35920756relu_dense_qt_1_35920758*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_relu_dense_Qt_1_layer_call_and_return_conditional_losses_359205832)
'relu_dense_Qt_1/StatefulPartitionedCallç
'relu_dense_Qt_2/StatefulPartitionedCallStatefulPartitionedCall0relu_dense_Qt_1/StatefulPartitionedCall:output:0relu_dense_qt_2_35920761relu_dense_qt_2_35920763*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_relu_dense_Qt_2_layer_call_and_return_conditional_losses_359206102)
'relu_dense_Qt_2/StatefulPartitionedCallç
'relu_dense_Qt_3/StatefulPartitionedCallStatefulPartitionedCall0relu_dense_Qt_2/StatefulPartitionedCall:output:0relu_dense_qt_3_35920766relu_dense_qt_3_35920768*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_relu_dense_Qt_3_layer_call_and_return_conditional_losses_359206372)
'relu_dense_Qt_3/StatefulPartitionedCallâ
&Q_target_value/StatefulPartitionedCallStatefulPartitionedCall0relu_dense_Qt_3/StatefulPartitionedCall:output:0q_target_value_35920771q_target_value_35920773*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Q_target_value_layer_call_and_return_conditional_losses_359206642(
&Q_target_value/StatefulPartitionedCallª
IdentityIdentity/Q_target_value/StatefulPartitionedCall:output:0'^Q_target_value/StatefulPartitionedCall(^relu_dense_Qt_1/StatefulPartitionedCall(^relu_dense_Qt_2/StatefulPartitionedCall(^relu_dense_Qt_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::2P
&Q_target_value/StatefulPartitionedCall&Q_target_value/StatefulPartitionedCall2R
'relu_dense_Qt_1/StatefulPartitionedCall'relu_dense_Qt_1/StatefulPartitionedCall2R
'relu_dense_Qt_2/StatefulPartitionedCall'relu_dense_Qt_2/StatefulPartitionedCall2R
'relu_dense_Qt_3/StatefulPartitionedCall'relu_dense_Qt_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
µ
M__inference_relu_dense_Qt_1_layer_call_and_return_conditional_losses_35920936

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
Ý
/__inference_functional_3_layer_call_fn_35920796	
state
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_functional_3_layer_call_and_return_conditional_losses_359207772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate
¦
Ý
/__inference_functional_3_layer_call_fn_35920751	
state
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallstateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_functional_3_layer_call_and_return_conditional_losses_359207322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate
­
þ
J__inference_functional_3_layer_call_and_return_conditional_losses_35920732

inputs
relu_dense_qt_1_35920711
relu_dense_qt_1_35920713
relu_dense_qt_2_35920716
relu_dense_qt_2_35920718
relu_dense_qt_3_35920721
relu_dense_qt_3_35920723
q_target_value_35920726
q_target_value_35920728
identity¢&Q_target_value/StatefulPartitionedCall¢'relu_dense_Qt_1/StatefulPartitionedCall¢'relu_dense_Qt_2/StatefulPartitionedCall¢'relu_dense_Qt_3/StatefulPartitionedCall½
'relu_dense_Qt_1/StatefulPartitionedCallStatefulPartitionedCallinputsrelu_dense_qt_1_35920711relu_dense_qt_1_35920713*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_relu_dense_Qt_1_layer_call_and_return_conditional_losses_359205832)
'relu_dense_Qt_1/StatefulPartitionedCallç
'relu_dense_Qt_2/StatefulPartitionedCallStatefulPartitionedCall0relu_dense_Qt_1/StatefulPartitionedCall:output:0relu_dense_qt_2_35920716relu_dense_qt_2_35920718*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_relu_dense_Qt_2_layer_call_and_return_conditional_losses_359206102)
'relu_dense_Qt_2/StatefulPartitionedCallç
'relu_dense_Qt_3/StatefulPartitionedCallStatefulPartitionedCall0relu_dense_Qt_2/StatefulPartitionedCall:output:0relu_dense_qt_3_35920721relu_dense_qt_3_35920723*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_relu_dense_Qt_3_layer_call_and_return_conditional_losses_359206372)
'relu_dense_Qt_3/StatefulPartitionedCallâ
&Q_target_value/StatefulPartitionedCallStatefulPartitionedCall0relu_dense_Qt_3/StatefulPartitionedCall:output:0q_target_value_35920726q_target_value_35920728*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Q_target_value_layer_call_and_return_conditional_losses_359206642(
&Q_target_value/StatefulPartitionedCallª
IdentityIdentity/Q_target_value/StatefulPartitionedCall:output:0'^Q_target_value/StatefulPartitionedCall(^relu_dense_Qt_1/StatefulPartitionedCall(^relu_dense_Qt_2/StatefulPartitionedCall(^relu_dense_Qt_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::2P
&Q_target_value/StatefulPartitionedCall&Q_target_value/StatefulPartitionedCall2R
'relu_dense_Qt_1/StatefulPartitionedCall'relu_dense_Qt_1/StatefulPartitionedCall2R
'relu_dense_Qt_2/StatefulPartitionedCall'relu_dense_Qt_2/StatefulPartitionedCall2R
'relu_dense_Qt_3/StatefulPartitionedCall'relu_dense_Qt_3/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©
Þ
/__inference_functional_3_layer_call_fn_35920925

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÈ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_functional_3_layer_call_and_return_conditional_losses_359207772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ"

J__inference_functional_3_layer_call_and_return_conditional_losses_35920851

inputs2
.relu_dense_qt_1_matmul_readvariableop_resource3
/relu_dense_qt_1_biasadd_readvariableop_resource2
.relu_dense_qt_2_matmul_readvariableop_resource3
/relu_dense_qt_2_biasadd_readvariableop_resource2
.relu_dense_qt_3_matmul_readvariableop_resource3
/relu_dense_qt_3_biasadd_readvariableop_resource1
-q_target_value_matmul_readvariableop_resource2
.q_target_value_biasadd_readvariableop_resource
identity½
%relu_dense_Qt_1/MatMul/ReadVariableOpReadVariableOp.relu_dense_qt_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02'
%relu_dense_Qt_1/MatMul/ReadVariableOp£
relu_dense_Qt_1/MatMulMatMulinputs-relu_dense_Qt_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
relu_dense_Qt_1/MatMul¼
&relu_dense_Qt_1/BiasAdd/ReadVariableOpReadVariableOp/relu_dense_qt_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&relu_dense_Qt_1/BiasAdd/ReadVariableOpÁ
relu_dense_Qt_1/BiasAddBiasAdd relu_dense_Qt_1/MatMul:product:0.relu_dense_Qt_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
relu_dense_Qt_1/BiasAdd
relu_dense_Qt_1/ReluRelu relu_dense_Qt_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
relu_dense_Qt_1/Relu½
%relu_dense_Qt_2/MatMul/ReadVariableOpReadVariableOp.relu_dense_qt_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02'
%relu_dense_Qt_2/MatMul/ReadVariableOp¿
relu_dense_Qt_2/MatMulMatMul"relu_dense_Qt_1/Relu:activations:0-relu_dense_Qt_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
relu_dense_Qt_2/MatMul¼
&relu_dense_Qt_2/BiasAdd/ReadVariableOpReadVariableOp/relu_dense_qt_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&relu_dense_Qt_2/BiasAdd/ReadVariableOpÁ
relu_dense_Qt_2/BiasAddBiasAdd relu_dense_Qt_2/MatMul:product:0.relu_dense_Qt_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
relu_dense_Qt_2/BiasAdd
relu_dense_Qt_2/ReluRelu relu_dense_Qt_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
relu_dense_Qt_2/Relu½
%relu_dense_Qt_3/MatMul/ReadVariableOpReadVariableOp.relu_dense_qt_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02'
%relu_dense_Qt_3/MatMul/ReadVariableOp¿
relu_dense_Qt_3/MatMulMatMul"relu_dense_Qt_2/Relu:activations:0-relu_dense_Qt_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
relu_dense_Qt_3/MatMul¼
&relu_dense_Qt_3/BiasAdd/ReadVariableOpReadVariableOp/relu_dense_qt_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&relu_dense_Qt_3/BiasAdd/ReadVariableOpÁ
relu_dense_Qt_3/BiasAddBiasAdd relu_dense_Qt_3/MatMul:product:0.relu_dense_Qt_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
relu_dense_Qt_3/BiasAdd
relu_dense_Qt_3/ReluRelu relu_dense_Qt_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
relu_dense_Qt_3/Reluº
$Q_target_value/MatMul/ReadVariableOpReadVariableOp-q_target_value_matmul_readvariableop_resource*
_output_shapes

: *
dtype02&
$Q_target_value/MatMul/ReadVariableOp¼
Q_target_value/MatMulMatMul"relu_dense_Qt_3/Relu:activations:0,Q_target_value/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Q_target_value/MatMul¹
%Q_target_value/BiasAdd/ReadVariableOpReadVariableOp.q_target_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Q_target_value/BiasAdd/ReadVariableOp½
Q_target_value/BiasAddBiasAddQ_target_value/MatMul:product:0-Q_target_value/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Q_target_value/BiasAdd
Q_target_value/ReluReluQ_target_value/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Q_target_value/Reluu
IdentityIdentity!Q_target_value/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ:::::::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
µ
M__inference_relu_dense_Qt_1_layer_call_and_return_conditional_losses_35920583

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ"

J__inference_functional_3_layer_call_and_return_conditional_losses_35920883

inputs2
.relu_dense_qt_1_matmul_readvariableop_resource3
/relu_dense_qt_1_biasadd_readvariableop_resource2
.relu_dense_qt_2_matmul_readvariableop_resource3
/relu_dense_qt_2_biasadd_readvariableop_resource2
.relu_dense_qt_3_matmul_readvariableop_resource3
/relu_dense_qt_3_biasadd_readvariableop_resource1
-q_target_value_matmul_readvariableop_resource2
.q_target_value_biasadd_readvariableop_resource
identity½
%relu_dense_Qt_1/MatMul/ReadVariableOpReadVariableOp.relu_dense_qt_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02'
%relu_dense_Qt_1/MatMul/ReadVariableOp£
relu_dense_Qt_1/MatMulMatMulinputs-relu_dense_Qt_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
relu_dense_Qt_1/MatMul¼
&relu_dense_Qt_1/BiasAdd/ReadVariableOpReadVariableOp/relu_dense_qt_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&relu_dense_Qt_1/BiasAdd/ReadVariableOpÁ
relu_dense_Qt_1/BiasAddBiasAdd relu_dense_Qt_1/MatMul:product:0.relu_dense_Qt_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
relu_dense_Qt_1/BiasAdd
relu_dense_Qt_1/ReluRelu relu_dense_Qt_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
relu_dense_Qt_1/Relu½
%relu_dense_Qt_2/MatMul/ReadVariableOpReadVariableOp.relu_dense_qt_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype02'
%relu_dense_Qt_2/MatMul/ReadVariableOp¿
relu_dense_Qt_2/MatMulMatMul"relu_dense_Qt_1/Relu:activations:0-relu_dense_Qt_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
relu_dense_Qt_2/MatMul¼
&relu_dense_Qt_2/BiasAdd/ReadVariableOpReadVariableOp/relu_dense_qt_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&relu_dense_Qt_2/BiasAdd/ReadVariableOpÁ
relu_dense_Qt_2/BiasAddBiasAdd relu_dense_Qt_2/MatMul:product:0.relu_dense_Qt_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
relu_dense_Qt_2/BiasAdd
relu_dense_Qt_2/ReluRelu relu_dense_Qt_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
relu_dense_Qt_2/Relu½
%relu_dense_Qt_3/MatMul/ReadVariableOpReadVariableOp.relu_dense_qt_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype02'
%relu_dense_Qt_3/MatMul/ReadVariableOp¿
relu_dense_Qt_3/MatMulMatMul"relu_dense_Qt_2/Relu:activations:0-relu_dense_Qt_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
relu_dense_Qt_3/MatMul¼
&relu_dense_Qt_3/BiasAdd/ReadVariableOpReadVariableOp/relu_dense_qt_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&relu_dense_Qt_3/BiasAdd/ReadVariableOpÁ
relu_dense_Qt_3/BiasAddBiasAdd relu_dense_Qt_3/MatMul:product:0.relu_dense_Qt_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
relu_dense_Qt_3/BiasAdd
relu_dense_Qt_3/ReluRelu relu_dense_Qt_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
relu_dense_Qt_3/Reluº
$Q_target_value/MatMul/ReadVariableOpReadVariableOp-q_target_value_matmul_readvariableop_resource*
_output_shapes

: *
dtype02&
$Q_target_value/MatMul/ReadVariableOp¼
Q_target_value/MatMulMatMul"relu_dense_Qt_3/Relu:activations:0,Q_target_value/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Q_target_value/MatMul¹
%Q_target_value/BiasAdd/ReadVariableOpReadVariableOp.q_target_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%Q_target_value/BiasAdd/ReadVariableOp½
Q_target_value/BiasAddBiasAddQ_target_value/MatMul:product:0-Q_target_value/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Q_target_value/BiasAdd
Q_target_value/ReluReluQ_target_value/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Q_target_value/Reluu
IdentityIdentity!Q_target_value/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ:::::::::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
´
L__inference_Q_target_value_layer_call_and_return_conditional_losses_35920664

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
é

!__inference__traced_save_35921052
file_prefix5
1savev2_relu_dense_qt_1_kernel_read_readvariableop3
/savev2_relu_dense_qt_1_bias_read_readvariableop5
1savev2_relu_dense_qt_2_kernel_read_readvariableop3
/savev2_relu_dense_qt_2_bias_read_readvariableop5
1savev2_relu_dense_qt_3_kernel_read_readvariableop3
/savev2_relu_dense_qt_3_bias_read_readvariableop4
0savev2_q_target_value_kernel_read_readvariableop2
.savev2_q_target_value_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b5d53da9b4ec4bd5832297aba329823c/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÙ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*ë
valueáBÞ	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slicesÐ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_relu_dense_qt_1_kernel_read_readvariableop/savev2_relu_dense_qt_1_bias_read_readvariableop1savev2_relu_dense_qt_2_kernel_read_readvariableop/savev2_relu_dense_qt_2_bias_read_readvariableop1savev2_relu_dense_qt_3_kernel_read_readvariableop/savev2_relu_dense_qt_3_bias_read_readvariableop0savev2_q_target_value_kernel_read_readvariableop.savev2_q_target_value_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*W
_input_shapesF
D: :@:@:@@:@:@ : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::	

_output_shapes
: 
*
Ë
#__inference__wrapped_model_35920568	
state?
;functional_3_relu_dense_qt_1_matmul_readvariableop_resource@
<functional_3_relu_dense_qt_1_biasadd_readvariableop_resource?
;functional_3_relu_dense_qt_2_matmul_readvariableop_resource@
<functional_3_relu_dense_qt_2_biasadd_readvariableop_resource?
;functional_3_relu_dense_qt_3_matmul_readvariableop_resource@
<functional_3_relu_dense_qt_3_biasadd_readvariableop_resource>
:functional_3_q_target_value_matmul_readvariableop_resource?
;functional_3_q_target_value_biasadd_readvariableop_resource
identityä
2functional_3/relu_dense_Qt_1/MatMul/ReadVariableOpReadVariableOp;functional_3_relu_dense_qt_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype024
2functional_3/relu_dense_Qt_1/MatMul/ReadVariableOpÉ
#functional_3/relu_dense_Qt_1/MatMulMatMulstate:functional_3/relu_dense_Qt_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#functional_3/relu_dense_Qt_1/MatMulã
3functional_3/relu_dense_Qt_1/BiasAdd/ReadVariableOpReadVariableOp<functional_3_relu_dense_qt_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3functional_3/relu_dense_Qt_1/BiasAdd/ReadVariableOpõ
$functional_3/relu_dense_Qt_1/BiasAddBiasAdd-functional_3/relu_dense_Qt_1/MatMul:product:0;functional_3/relu_dense_Qt_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$functional_3/relu_dense_Qt_1/BiasAdd¯
!functional_3/relu_dense_Qt_1/ReluRelu-functional_3/relu_dense_Qt_1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!functional_3/relu_dense_Qt_1/Reluä
2functional_3/relu_dense_Qt_2/MatMul/ReadVariableOpReadVariableOp;functional_3_relu_dense_qt_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype024
2functional_3/relu_dense_Qt_2/MatMul/ReadVariableOpó
#functional_3/relu_dense_Qt_2/MatMulMatMul/functional_3/relu_dense_Qt_1/Relu:activations:0:functional_3/relu_dense_Qt_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2%
#functional_3/relu_dense_Qt_2/MatMulã
3functional_3/relu_dense_Qt_2/BiasAdd/ReadVariableOpReadVariableOp<functional_3_relu_dense_qt_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3functional_3/relu_dense_Qt_2/BiasAdd/ReadVariableOpõ
$functional_3/relu_dense_Qt_2/BiasAddBiasAdd-functional_3/relu_dense_Qt_2/MatMul:product:0;functional_3/relu_dense_Qt_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2&
$functional_3/relu_dense_Qt_2/BiasAdd¯
!functional_3/relu_dense_Qt_2/ReluRelu-functional_3/relu_dense_Qt_2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2#
!functional_3/relu_dense_Qt_2/Reluä
2functional_3/relu_dense_Qt_3/MatMul/ReadVariableOpReadVariableOp;functional_3_relu_dense_qt_3_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype024
2functional_3/relu_dense_Qt_3/MatMul/ReadVariableOpó
#functional_3/relu_dense_Qt_3/MatMulMatMul/functional_3/relu_dense_Qt_2/Relu:activations:0:functional_3/relu_dense_Qt_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2%
#functional_3/relu_dense_Qt_3/MatMulã
3functional_3/relu_dense_Qt_3/BiasAdd/ReadVariableOpReadVariableOp<functional_3_relu_dense_qt_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype025
3functional_3/relu_dense_Qt_3/BiasAdd/ReadVariableOpõ
$functional_3/relu_dense_Qt_3/BiasAddBiasAdd-functional_3/relu_dense_Qt_3/MatMul:product:0;functional_3/relu_dense_Qt_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2&
$functional_3/relu_dense_Qt_3/BiasAdd¯
!functional_3/relu_dense_Qt_3/ReluRelu-functional_3/relu_dense_Qt_3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2#
!functional_3/relu_dense_Qt_3/Reluá
1functional_3/Q_target_value/MatMul/ReadVariableOpReadVariableOp:functional_3_q_target_value_matmul_readvariableop_resource*
_output_shapes

: *
dtype023
1functional_3/Q_target_value/MatMul/ReadVariableOpð
"functional_3/Q_target_value/MatMulMatMul/functional_3/relu_dense_Qt_3/Relu:activations:09functional_3/Q_target_value/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"functional_3/Q_target_value/MatMulà
2functional_3/Q_target_value/BiasAdd/ReadVariableOpReadVariableOp;functional_3_q_target_value_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2functional_3/Q_target_value/BiasAdd/ReadVariableOpñ
#functional_3/Q_target_value/BiasAddBiasAdd,functional_3/Q_target_value/MatMul:product:0:functional_3/Q_target_value/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#functional_3/Q_target_value/BiasAdd¬
 functional_3/Q_target_value/ReluRelu,functional_3/Q_target_value/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 functional_3/Q_target_value/Relu
IdentityIdentity.functional_3/Q_target_value/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ:::::::::N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate
ª
ý
J__inference_functional_3_layer_call_and_return_conditional_losses_35920705	
state
relu_dense_qt_1_35920684
relu_dense_qt_1_35920686
relu_dense_qt_2_35920689
relu_dense_qt_2_35920691
relu_dense_qt_3_35920694
relu_dense_qt_3_35920696
q_target_value_35920699
q_target_value_35920701
identity¢&Q_target_value/StatefulPartitionedCall¢'relu_dense_Qt_1/StatefulPartitionedCall¢'relu_dense_Qt_2/StatefulPartitionedCall¢'relu_dense_Qt_3/StatefulPartitionedCall¼
'relu_dense_Qt_1/StatefulPartitionedCallStatefulPartitionedCallstaterelu_dense_qt_1_35920684relu_dense_qt_1_35920686*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_relu_dense_Qt_1_layer_call_and_return_conditional_losses_359205832)
'relu_dense_Qt_1/StatefulPartitionedCallç
'relu_dense_Qt_2/StatefulPartitionedCallStatefulPartitionedCall0relu_dense_Qt_1/StatefulPartitionedCall:output:0relu_dense_qt_2_35920689relu_dense_qt_2_35920691*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_relu_dense_Qt_2_layer_call_and_return_conditional_losses_359206102)
'relu_dense_Qt_2/StatefulPartitionedCallç
'relu_dense_Qt_3/StatefulPartitionedCallStatefulPartitionedCall0relu_dense_Qt_2/StatefulPartitionedCall:output:0relu_dense_qt_3_35920694relu_dense_qt_3_35920696*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_relu_dense_Qt_3_layer_call_and_return_conditional_losses_359206372)
'relu_dense_Qt_3/StatefulPartitionedCallâ
&Q_target_value/StatefulPartitionedCallStatefulPartitionedCall0relu_dense_Qt_3/StatefulPartitionedCall:output:0q_target_value_35920699q_target_value_35920701*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Q_target_value_layer_call_and_return_conditional_losses_359206642(
&Q_target_value/StatefulPartitionedCallª
IdentityIdentity/Q_target_value/StatefulPartitionedCall:output:0'^Q_target_value/StatefulPartitionedCall(^relu_dense_Qt_1/StatefulPartitionedCall(^relu_dense_Qt_2/StatefulPartitionedCall(^relu_dense_Qt_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::2P
&Q_target_value/StatefulPartitionedCall&Q_target_value/StatefulPartitionedCall2R
'relu_dense_Qt_1/StatefulPartitionedCall'relu_dense_Qt_1/StatefulPartitionedCall2R
'relu_dense_Qt_2/StatefulPartitionedCall'relu_dense_Qt_2/StatefulPartitionedCall2R
'relu_dense_Qt_3/StatefulPartitionedCall'relu_dense_Qt_3/StatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate
ª
ý
J__inference_functional_3_layer_call_and_return_conditional_losses_35920681	
state
relu_dense_qt_1_35920594
relu_dense_qt_1_35920596
relu_dense_qt_2_35920621
relu_dense_qt_2_35920623
relu_dense_qt_3_35920648
relu_dense_qt_3_35920650
q_target_value_35920675
q_target_value_35920677
identity¢&Q_target_value/StatefulPartitionedCall¢'relu_dense_Qt_1/StatefulPartitionedCall¢'relu_dense_Qt_2/StatefulPartitionedCall¢'relu_dense_Qt_3/StatefulPartitionedCall¼
'relu_dense_Qt_1/StatefulPartitionedCallStatefulPartitionedCallstaterelu_dense_qt_1_35920594relu_dense_qt_1_35920596*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_relu_dense_Qt_1_layer_call_and_return_conditional_losses_359205832)
'relu_dense_Qt_1/StatefulPartitionedCallç
'relu_dense_Qt_2/StatefulPartitionedCallStatefulPartitionedCall0relu_dense_Qt_1/StatefulPartitionedCall:output:0relu_dense_qt_2_35920621relu_dense_qt_2_35920623*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_relu_dense_Qt_2_layer_call_and_return_conditional_losses_359206102)
'relu_dense_Qt_2/StatefulPartitionedCallç
'relu_dense_Qt_3/StatefulPartitionedCallStatefulPartitionedCall0relu_dense_Qt_2/StatefulPartitionedCall:output:0relu_dense_qt_3_35920648relu_dense_qt_3_35920650*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_relu_dense_Qt_3_layer_call_and_return_conditional_losses_359206372)
'relu_dense_Qt_3/StatefulPartitionedCallâ
&Q_target_value/StatefulPartitionedCallStatefulPartitionedCall0relu_dense_Qt_3/StatefulPartitionedCall:output:0q_target_value_35920675q_target_value_35920677*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_Q_target_value_layer_call_and_return_conditional_losses_359206642(
&Q_target_value/StatefulPartitionedCallª
IdentityIdentity/Q_target_value/StatefulPartitionedCall:output:0'^Q_target_value/StatefulPartitionedCall(^relu_dense_Qt_1/StatefulPartitionedCall(^relu_dense_Qt_2/StatefulPartitionedCall(^relu_dense_Qt_3/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:ÿÿÿÿÿÿÿÿÿ::::::::2P
&Q_target_value/StatefulPartitionedCall&Q_target_value/StatefulPartitionedCall2R
'relu_dense_Qt_1/StatefulPartitionedCall'relu_dense_Qt_1/StatefulPartitionedCall2R
'relu_dense_Qt_2/StatefulPartitionedCall'relu_dense_Qt_2/StatefulPartitionedCall2R
'relu_dense_Qt_3/StatefulPartitionedCall'relu_dense_Qt_3/StatefulPartitionedCall:N J
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

_user_specified_namestate"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*­
serving_default
7
state.
serving_default_state:0ÿÿÿÿÿÿÿÿÿB
Q_target_value0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:
¾-
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	optimizer
loss
regularization_losses
	trainable_variables

	variables
	keras_api

signatures
>__call__
*?&call_and_return_all_conditional_losses
@_default_save_signature"À*
_tf_keras_network¤*{"class_name": "Functional", "name": "functional_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "state"}, "name": "state", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "relu_dense_Qt_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "relu_dense_Qt_1", "inbound_nodes": [[["state", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "relu_dense_Qt_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "relu_dense_Qt_2", "inbound_nodes": [[["relu_dense_Qt_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "relu_dense_Qt_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "relu_dense_Qt_3", "inbound_nodes": [[["relu_dense_Qt_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Q_target_value", "trainable": true, "dtype": "float32", "units": 2, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Q_target_value", "inbound_nodes": [[["relu_dense_Qt_3", 0, 0, {}]]]}], "input_layers": [["state", 0, 0]], "output_layers": [["Q_target_value", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "state"}, "name": "state", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "relu_dense_Qt_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "relu_dense_Qt_1", "inbound_nodes": [[["state", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "relu_dense_Qt_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "relu_dense_Qt_2", "inbound_nodes": [[["relu_dense_Qt_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "relu_dense_Qt_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "relu_dense_Qt_3", "inbound_nodes": [[["relu_dense_Qt_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "Q_target_value", "trainable": true, "dtype": "float32", "units": 2, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "Q_target_value", "inbound_nodes": [[["relu_dense_Qt_3", 0, 0, {}]]]}], "input_layers": [["state", 0, 0]], "output_layers": [["Q_target_value", 0, 0]]}}, "training_config": {"loss": ["mse"], "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
å"â
_tf_keras_input_layerÂ{"class_name": "InputLayer", "name": "state", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "state"}}
ý

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
A__call__
*B&call_and_return_all_conditional_losses"Ø
_tf_keras_layer¾{"class_name": "Dense", "name": "relu_dense_Qt_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu_dense_Qt_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
ÿ

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
C__call__
*D&call_and_return_all_conditional_losses"Ú
_tf_keras_layerÀ{"class_name": "Dense", "name": "relu_dense_Qt_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu_dense_Qt_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ÿ

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
E__call__
*F&call_and_return_all_conditional_losses"Ú
_tf_keras_layerÀ{"class_name": "Dense", "name": "relu_dense_Qt_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "relu_dense_Qt_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ü

kernel
 bias
!regularization_losses
"trainable_variables
#	variables
$	keras_api
G__call__
*H&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Dense", "name": "Q_target_value", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Q_target_value", "trainable": true, "dtype": "float32", "units": 2, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
"
	optimizer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
 7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
 7"
trackable_list_wrapper
Ê
regularization_losses
%metrics
&non_trainable_variables
'layer_metrics

(layers
	trainable_variables

	variables
)layer_regularization_losses
>__call__
@_default_save_signature
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
,
Iserving_default"
signature_map
(:&@2relu_dense_Qt_1/kernel
": @2relu_dense_Qt_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
*metrics
+non_trainable_variables
,layer_metrics

-layers
trainable_variables
	variables
.layer_regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
(:&@@2relu_dense_Qt_2/kernel
": @2relu_dense_Qt_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
/metrics
0non_trainable_variables
1layer_metrics

2layers
trainable_variables
	variables
3layer_regularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
(:&@ 2relu_dense_Qt_3/kernel
":  2relu_dense_Qt_3/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
4metrics
5non_trainable_variables
6layer_metrics

7layers
trainable_variables
	variables
8layer_regularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
':% 2Q_target_value/kernel
!:2Q_target_value/bias
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
­
!regularization_losses
9metrics
:non_trainable_variables
;layer_metrics

<layers
"trainable_variables
#	variables
=layer_regularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
2
/__inference_functional_3_layer_call_fn_35920751
/__inference_functional_3_layer_call_fn_35920925
/__inference_functional_3_layer_call_fn_35920904
/__inference_functional_3_layer_call_fn_35920796À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ö2ó
J__inference_functional_3_layer_call_and_return_conditional_losses_35920851
J__inference_functional_3_layer_call_and_return_conditional_losses_35920681
J__inference_functional_3_layer_call_and_return_conditional_losses_35920883
J__inference_functional_3_layer_call_and_return_conditional_losses_35920705À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ß2Ü
#__inference__wrapped_model_35920568´
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *$¢!

stateÿÿÿÿÿÿÿÿÿ
Ü2Ù
2__inference_relu_dense_Qt_1_layer_call_fn_35920945¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_relu_dense_Qt_1_layer_call_and_return_conditional_losses_35920936¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ü2Ù
2__inference_relu_dense_Qt_2_layer_call_fn_35920965¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_relu_dense_Qt_2_layer_call_and_return_conditional_losses_35920956¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ü2Ù
2__inference_relu_dense_Qt_3_layer_call_fn_35920985¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_relu_dense_Qt_3_layer_call_and_return_conditional_losses_35920976¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Û2Ø
1__inference_Q_target_value_layer_call_fn_35921005¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ö2ó
L__inference_Q_target_value_layer_call_and_return_conditional_losses_35920996¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
3B1
&__inference_signature_wrapper_35920819state¬
L__inference_Q_target_value_layer_call_and_return_conditional_losses_35920996\ /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_Q_target_value_layer_call_fn_35921005O /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ¢
#__inference__wrapped_model_35920568{ .¢+
$¢!

stateÿÿÿÿÿÿÿÿÿ
ª "?ª<
:
Q_target_value(%
Q_target_valueÿÿÿÿÿÿÿÿÿ·
J__inference_functional_3_layer_call_and_return_conditional_losses_35920681i 6¢3
,¢)

stateÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
J__inference_functional_3_layer_call_and_return_conditional_losses_35920705i 6¢3
,¢)

stateÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
J__inference_functional_3_layer_call_and_return_conditional_losses_35920851j 7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
J__inference_functional_3_layer_call_and_return_conditional_losses_35920883j 7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_functional_3_layer_call_fn_35920751\ 6¢3
,¢)

stateÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_functional_3_layer_call_fn_35920796\ 6¢3
,¢)

stateÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_functional_3_layer_call_fn_35920904] 7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_functional_3_layer_call_fn_35920925] 7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ­
M__inference_relu_dense_Qt_1_layer_call_and_return_conditional_losses_35920936\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
2__inference_relu_dense_Qt_1_layer_call_fn_35920945O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@­
M__inference_relu_dense_Qt_2_layer_call_and_return_conditional_losses_35920956\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
2__inference_relu_dense_Qt_2_layer_call_fn_35920965O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ@­
M__inference_relu_dense_Qt_3_layer_call_and_return_conditional_losses_35920976\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
2__inference_relu_dense_Qt_3_layer_call_fn_35920985O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ ¯
&__inference_signature_wrapper_35920819 7¢4
¢ 
-ª*
(
state
stateÿÿÿÿÿÿÿÿÿ"?ª<
:
Q_target_value(%
Q_target_valueÿÿÿÿÿÿÿÿÿ