//
//  DenseLayerModel.swift
//  Iris
//
//  Created by Jean Flaherty on 12/1/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import Foundation
import TensorFlow

struct DenseLayerModel<Scalar: TensorFlowScalar & BinaryFloatingPoint> : InferenceModel where Scalar.RawSignificand : FixedWidthInteger {
    typealias Parameter = DenseLayer<Scalar>.Parameter
    typealias DenseLayersConfig = (size: Int32, activation: DenseLayer<Scalar>.ActivationFunction)
    // Layers
    var layers = [DenseLayer<Scalar>]()
    
    @inlinable @inline(__always)
    init(inputSize: Int32, layersConfig: [DenseLayersConfig]) {
        guard let firstLayerConfig = layersConfig.first else { return }
        let firstLayer = DenseLayer<Scalar>(inputSize: inputSize, outputSize: firstLayerConfig.size, activation: firstLayerConfig.activation)
        layers.append(firstLayer)
        
        for i in 0..<layersConfig.count-1 {
            let inputSize = layersConfig[i].size
            let outputSize = layersConfig[i+1].size
            let activation = layersConfig[i+1].activation
            let layer = DenseLayer<Scalar>(inputSize: inputSize, outputSize: outputSize, activation: activation)
            layers.append(layer)
        }
    }
    
    @inlinable @inline(__always)
    func inference(_ x: Tensor<Scalar>) -> Tensor<Scalar> {
        return layers.reduce(x) { $1.activate($0) }
    }
    
}

