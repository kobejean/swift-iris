//
//  IrisDataset.swift
//  Iris
//
//  Created by Jean Flaherty on 12/7/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import Foundation
import TensorFlow

func readIrisDataset() -> ([Float], [Int32], Int32) {
    let fileURLString = "usr/share/Iris/Data/iris_training.csv"
    guard let trainingData = try? String(contentsOfFile: fileURLString) else {
        fatalError("Could not read file at \(fileURLString)")
    }
    let trainingDataArray = trainingData.split(separator: "\n")
        .map { $0.split(separator: ",").map { String($0) } }
        .dropFirst()
    let (featureArray, labelArray) = trainingDataArray.reduce(into: ([Float](), [Int32]())) { (result, row) in
        result.0 += row[0..<4].map { Float($0)! }
        result.1.append(Int32(row[4])!)
    }
    let rowCount = Int32(trainingDataArray.count)
    return (featureArray, labelArray, rowCount)
}
