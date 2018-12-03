//
//  main.swift
//  Iris
//
//  Created by Jean Flaherty on 12/1/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import Foundation
import TensorFlow

let fileManager = FileManager.default

let fileURLString = "/Users/kobejean/Developer/git/swift-iris/Iris/Iris/Data/iris_training.csv"
guard var trainingData = try? String(contentsOfFile: "/Users/kobejean/Developer/git/swift-iris/Iris/Iris/Data/iris_training.csv") else {
    fatalError()
}
let trainingDataArray = trainingData.split(separator: "\n")
                                    .map { $0.split(separator: ",").map { String($0) } }
                                    .dropFirst()
let (featureArray, labelArray) = trainingDataArray.reduce(into: ([Float](), [Int32]())) { (result, row) in
    result.0 += row[0..<4].map { Float($0)! }
    result.1.append(Int32(row[4])!)
}
let rowCount = Int32(trainingDataArray.count)
let features = Tensor<Float>(shape: [rowCount, 4], scalars: featureArray)
let labels = Tensor<Int32>(shape: [rowCount], scalars: labelArray)

let batchSize = Int32(rowCount)
let inputSize = Int32(4)
let layer1Size = Int32(5)
let layer2Size = Int32(5)
let outputSize = Int32(3)

//public struct IrisModel : ParameterGroup {
//    struct Layer : ParameterGroup {
//        let w: Tensor<Float>
//        let b: Tensor<Float>
//    }
//    public var layer1 =        DenseLayer<Float>(inputSize: inputSize,  outputSize: layer1Size, activation: relu)
//    public var layer2 =        DenseLayer<Float>(inputSize: layer1Size, outputSize: layer2Size, activation: relu)
//    public var outputLayer =   DenseLayer<Float>(inputSize: layer2Size, outputSize: outputSize, activation: softmax)
//}

public struct Layer : ParameterGroup {
    public var w: Tensor<Float>
    public var b: Tensor<Float>
    
    @inlinable @inline(__always)
    public init(inputSize: Int32,  outputSize: Int32) {
        w = Tensor<Float>(glorotUniform: [inputSize, outputSize])
        b = Tensor<Float>(zeros: [outputSize])
    }
}

public struct IrisModel : ParameterGroup {
    public var layer1 =        Layer(inputSize: inputSize,  outputSize: layer1Size)
    public var layer2 =        Layer(inputSize: layer1Size, outputSize: layer2Size)
    public var outputLayer =   Layer(inputSize: layer2Size, outputSize: outputSize)
}


@inlinable @inline(__always)
public func inference(_ x: Tensor<Float>, model: IrisModel) -> Tensor<Float> {
    let h1 = relu(x • model.layer1.w + model.layer1.b)
    let h2 = relu(h1 • model.layer2.w + model.layer2.b)
    let output = softmax(h2 • model.outputLayer.w + model.outputLayer.b)
    return output
}

let model = IrisModel()

@inlinable @inline(__always)
func loss(_ x: Tensor<Float>, using model: IrisModel, labels: Tensor<Int32>) -> Float {
    let logits = inference(x, model: model)
    let oneHotLabels: Tensor<Float> = oneHot(indices: labels, depth: 3, onValue: -1, offValue: 0)
    let trueLogProb = oneHotLabels • log(logits).transposed()
    let total_loss = trueLogProb.sum(squeezingAxes: 1).mean()
    return total_loss
}

let total_loss = loss(features, using: model, labels: labels)
//print(#valueAndGradient(relu(_:))(features))
//print(#valueAndGradient(loss(_:using:labels:))(features, model, labels))
print("Loss:", total_loss)
