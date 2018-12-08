//
//  IrisModel.swift
//  Iris
//
//  Created by Jean Flaherty on 12/7/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow

public struct IrisModel : ParameterGroup {
    public typealias Parameter = Tensor<Float>
    public var w1: Parameter
    public var w2: Parameter
    public var w3: Parameter
    public var b1: Parameter
    public var b2: Parameter
    public var b3: Parameter
    
    @inlinable @inline(__always)
    public init() {
        w1 = Parameter(glorotUniform: [inputSize, layer1Size])
        w2 = Parameter(glorotUniform: [layer1Size, layer2Size])
        w3 = Parameter(glorotUniform: [layer2Size, outputSize])
        b1 = Parameter(zeros: [layer1Size])
        b2 = Parameter(zeros: [layer2Size])
        b3 = Parameter(zeros: [outputSize])
    }
    
    @inlinable @inline(__always)
    public init(w1: Parameter, w2: Parameter, w3: Parameter, b1: Parameter, b2: Parameter, b3: Parameter) {
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
    }
}
