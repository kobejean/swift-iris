//
//  DenseLayer.swift
//  Iris
//
//  Created by Jean Flaherty on 12/1/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import Foundation
import TensorFlow

public struct DenseLayer<Scalar: TensorFlowScalar & BinaryFloatingPoint> : ParameterGroup where Scalar.RawSignificand : FixedWidthInteger {
    public typealias Parameter = Tensor<Scalar>
    public typealias ActivationFunction = (Parameter) -> Parameter
    
    public var w: Parameter
    public var b: Parameter
    public var σ: ActivationFunction
    
    @inlinable @inline(__always)
    public init(inputSize: Int32, outputSize: Int32, activation: @escaping ActivationFunction) {
        w = Parameter(glorotUniform: [inputSize, outputSize])
        b = Parameter(zeros: [outputSize])
        σ = activation
    }
    
    @inlinable @inline(__always)
    public func activate(_ x: Parameter) -> Parameter {
        return σ(x • w + b)
    }
    
    public mutating func update(withGradients gradients: DenseLayer<Scalar>, _ updater: (inout Tensor<Scalar>, Tensor<Scalar>) -> Void) {
        updater(&w, gradients.w)
        updater(&b, gradients.b)
    }
}

