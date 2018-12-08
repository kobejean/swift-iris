# swift-iris
Train on the Iris dataset with swift.


## Requirements
* macOS 10.13.5 or later, with Xcode 10.0 or later; OR
* Ubuntu 16.04 (64-bit); OR
* other operating systems may work, but you will have to build Swift from
  sources.

This repository contains TensorFlow models written in Swift.

The `stable` branch works with the latest [Swift for TensorFlow prebuilt packages](https://github.com/tensorflow/swift/blob/master/Installation.md#pre-built-packages).

## Code

```swift
// Hyperparameters
public let inputSize = Int32(4)
public let layer1Size = Int32(10)
public let layer2Size = Int32(10)
public let outputSize = Int32(3)
let x = Tensor<Float>(shape: [batchSize, 4], scalars: featureArray)
let y_i = Tensor<Int32>(shape: [batchSize], scalars: labelArray)
var θ = IrisModel()
let η: Float = 0.001

@inlinable @inline(__always)
func lossAndGradient(_ x: Tensor<Float>, _ y_i: Tensor<Int32>, _ θ: IrisModel) -> (Float, IrisModel) {
    // Forward pass
    let z1 = x • θ.w1 + θ.b1
    let h1 = relu(z1)
    let z2 = h1 • θ.w2 + θ.b2
    let h2 = relu(z2)
    let z3 = h2 • θ.w3 + θ.b3
    let h3 = softmax(z3)
    
    // Evaluation
    let p = e_i(y_i, outputSize)
    let q = h3
    let H = -Σ(p * log(q))
    let H_total = μ(H)
    
    // Backward pass
    let dz3 = (q - p) / Float(batchSize)      // [B, l3]
    let dw3 = h2⊺ • dz3                       // [l2, l3]
    let db3 = Σ(dz3, 0)                       // [l3]
    let dz2 = dz3 • θ.w3⊺ * ajointRelu(z2)    // [B, l2]
    let dw2 = h1⊺ • dz2                       // [l1, l2]
    let db2 = Σ(dz2, 0)                       // [l2]
    let dz1 = dz2 • θ.w2⊺ * ajointRelu(z1)    // [B, l1]
    let dw1 = x⊺ • dz1                        // [x, l1]
    let db1 = Σ(dz1, 0)
    let dθ = IrisModel(w1: dw1, w2: dw2, w3: dw3, b1: db1, b2: db2, b3: db3)
    return (H_total, dθ)
}

while true {
    for i in 0..<100 {
        let (H_total, dθ) = lossAndGradient(x,y_i,θ)
        θ.update(withGradients: dθ) { θ_n, dθ_n in
            θ_n -= η * dθ_n
        }
        if i == 0 {
            // Print Loss and Likelihood
            print(String(format: "Training Loss: %.5f Likelihood: %.5f", H_total, exp(-H_total)))
        }
    }
}
```
