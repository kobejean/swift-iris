//
//  Ops.swift
//  Iris
//
//  Created by Jean Flaherty on 12/2/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow

@inlinable @inline(__always)
func oneHot<Scalar: TensorFlowScalar>(indices: Tensor<Int32>, depth: Int32, onValue: Scalar, offValue: Scalar, axis: Int64? = nil) -> Tensor<Scalar> {
    let depthTensor = Tensor<Int32>(depth)
    let onValueTensor = Tensor<Scalar>(onValue)
    let offValueTensor = Tensor<Scalar>(offValue)
    if let axis = axis {
        return Raw.oneHot(indices: indices, depth: depthTensor, onValue: onValueTensor, offValue: offValueTensor, axis: axis)
    }
    return Raw.oneHot(indices: indices, depth: depthTensor, onValue: onValueTensor, offValue: offValueTensor)
}

//@inlinable @inline(__always)
//@differentiable(reverse, wrt: (.0), adjoint: _adjointGather)
//func gather<Scalar: TensorFlowScalar>(_ x: Tensor<Scalar>, indices: Tensor<Int32>, axis: Int64? = nil) -> Tensor<Scalar> {
//    let axis = axis ?? Int64(x.shape.count-1)
//    let axisTensor = Tensor<Int64>(axis)
//    return Raw.gatherV2(params: x, indices: indices, axis: axisTensor)
//}
//
//func _adjointGather<Scalar: Numeric>(_ x: Tensor<Scalar>, indices: Tensor<Int32>, axis: Int64? = nil, originalResult: Tensor<Scalar>, seed: Tensor<Scalar>) -> Tensor<Scalar> {
//    let axis = axis ?? Int64(x.shape.count-1)
//    let depth = x.shape[axis]
//    let onValue: Scalar = 1
//    let offValue: Scalar = 0
//    let base = oneHot(indices: indices, depth: depth, onValue: onValue, offValue: offValue).broadcast()
//    return base
//}


@inlinable @inline(__always)
@differentiable(reverse, wrt: (.0), primal: _primalSoftmaxCrossEntropy, adjoint: _adjointSoftmaxCrossEntropy)
func softmaxCrossEntropy(logits: Tensor<Float>, categoricalLabels: Tensor<Int32>) -> Float {
    return Raw.sparseSoftmaxCrossEntropyWithLogits(features: logits,
                                                   labels: categoricalLabels).loss.mean()
}

@inlinable @inline(__always)
internal func _primalSoftmaxCrossEntropy(logits: Tensor<Float>,
                                         categoricalLabels: Tensor<Int32>) -> (Tensor<Float>, Float) {
    let (loss, grad) = Raw.sparseSoftmaxCrossEntropyWithLogits(features: logits,
                                                               labels: categoricalLabels)
    return (grad, loss.mean())
}

@inlinable @inline(__always)
internal func _adjointSoftmaxCrossEntropy(logits: Tensor<Float>,
                                          categoricalLabels: Tensor<Int32>,
                                          checkpointedGrad: Tensor<Float>,
                                          originalResult: Float,
                                          seed: Float) -> Tensor<Float> {
    return checkpointedGrad
}
