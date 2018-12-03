//
//  Model.swift
//  Iris
//
//  Created by Jean Flaherty on 12/2/18.
//  Copyright © 2018 Jean Flaherty. All rights reserved.
//

import TensorFlow

public protocol InferenceModel : ParameterGroup {
    associatedtype Scalar: TensorFlowScalar
    func inference(_: Tensor<Scalar>) -> Tensor<Scalar>
}
