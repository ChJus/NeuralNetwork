# Neural Network

Complete from-scratch implementation of neural networks (in ≈400-500 lines of code) with examples for training on the
MNIST handwritten digits dataset and CIFAR-10 object recognition dataset. See it in
action [here](https://chjus.github.io/NeuralNetwork/)!

## Implementation and Design

![](nn.png)

### Structure

A neural network consists of _layers_ of neurons. For example, in the diagram, there are three layers of neurons with 6,
4, and 3 neurons, respectively. Each neuron has weighted connections (whose value may signify the strength and
relationship of their connection) to each neuron in the next layer. Each neuron receives a
weighted input sum (denoted $\text{net}$, the sum of the products of corresponding weights and inputs), and applies an
_activation function_ $\varphi$ to constrain the weighted input within a certain fixed range. A popular choice
of $\varphi$ is the sigmoid activation function:

$$\sigma(x) = \frac{1}{1+e^{-x}},$$

which constrains values to be between 0 and 1 (useful, e.g., when you want the network to output a probability that an
input is from a classification class). You may refer to other activation functions in
the [corresponding section](#activation-functions) below.

The resulting activation becomes the input for the neurons in
the next layer. Notably, we denote each weight $w_{ij}$ connecting the $i\text{th}$ neuron in the current layer to
the $j\text{th}$ neuron in the next layer. As such, the input to the $j\text{th}$ neuron in the middle layer,

```math
o_{j} = \varphi(\text{net}_{j}) = \varphi\left(\sum_{k}{w_{kj}o_{k}}\right).
```

Note that each layer tends to have a _bias_ neuron, whose weight value is simply added to the weighted
sum $\text{net}$ (alternatively, you can consider its input to always be 1).

### Feedforward

In both the learning and classification stage, an example input array/vector is fed as input to the first layer, and
through the weighted sum and activation processes described previously, result in an output vector in the final layer.

### Backpropagation

For a network to learn, its adjustable parameters (the weights) are altered based on the network's classification
errors. Particularly, the goal in the learning process is to minimize error $E=L(t,y)$, where $L$ represents a _loss
function_ that computes error based on the desired target output $t$ and predicted output $y$.

For example, the partial derivative

```math
\frac{\partial E}{\partial w_{ij}}= \underbrace{\frac{\partial E}{\partial o_j} \frac{\partial o_j}{
\partial \text{net}_{j}}}_{\delta_j} \underbrace{\frac{\partial \text{net}_{j}}{\partial w_{ij}}}_{o_i}
```

represents the sensitivity of $E$ with respect to changes to $w_{ij}$. We update the weight $w_{ij}$ as

$$w_{ij} = w_{ij} + \Delta w_{ij},$$

with

$$\Delta w_{ij} = -\eta \frac{\partial E}{\partial w_{ij}}.$$

Notably, the negative sign ensures the weight is updated such that the error $E$ as caused by $w_{ij}$ is reduced. The
factor $\eta$ is referred to as _step size_ or _learning rate_, and controls the degree to which the weight $w_{ij}$ is
adjusted. Notably, a too large $\eta$ would lead to overcorrection of $w_{ij}$, which may lead to difficulty in
minimizing $E$ over all training examples, whereas a too small $\eta$ would result in minimal adjustments, leading to
slow learning.

We have

```math
\delta_j = \begin{cases}
\frac{\partial L(t, o_j)}{\partial o_j}\frac{d\varphi{(\text{net}_j)}}{d\text{net}_j} & \text{if }j \text{ is an output neuron.} \\
\left(\sum_{l\in L}{w_{jl}\cdot \delta_{l}}\right)\frac{d\varphi{(\text{net}_j)}}{d\text{net}_j} & \text{if }j \text{ is an input neuron.}
\end{cases}.
```

Correspondingly, in code:

```java
if (isOutputLayer) {
  for (int j = 0; j < weights[0].length; j++) {
    deltaWeights[j] = error[j] * activationFunction(weightedSumOutput[j], true);
  }
} else {
  for (int j = 0; j < nextLayer.weights.length; j++) {
    for (int l = 0; l < nextLayer.weights[0].length; l++) {
      deltaWeights[j] += nextLayer.weights[j][l] * nextLayer.deltaWeights[l];
    }
    deltaWeights[j] *= activationFunction(weightedSumOutput[j], true);
  }
}

for (int i = 0; i < weights.length; i++) {
  for (int j = 0; j < weights[i].length; j++) {
    weightsAdjustments[i][j] += deltaWeights[j] * inputs[i] * -learningRate;
  }
}

for (int j = 0; j < biases.length; j++) {
  biasesAdjustments[j] += deltaWeights[j] * -learningRate;
}
```

where `activationFunction(weightedSum, true)` finds the derivative $\frac{d\varphi{(\text{net}_j)}}{d\text{net}_j}$,
`weights` is a 2D array of size `[i][j]` and `biases` is an array of size `[j]`, for current layer with $i$ neurons and 
next layer with $j$ neurons.

### Loss functions

#### Mean-squared error

```math
L(t,y) = \sum_{i}(t_i-y_i)^2
```

```math
\frac{\partial L(t,y)}{\partial y_i} = 2(y_i - t_i)
```

#### Categorical cross-entropy

Used when examples can only be classified as one of several possible classification classes (one-hot encoding). 
Notably **used with `softmax` activation function**. Note that _binary cross-entropy_ with two classes is simply a special 
case of categorical cross-entropy 
([source](https://towardsdatascience.com/deriving-backpropagation-with-cross-entropy-loss-d24811edeaf9)). 

```math
L(t,y) = -\sum_{i}{t_i \log(y_i)}
```

```math
\frac{\partial L(t,y)}{\partial y_i} = -\frac{t_i}{y_i}
```

#### Multi-class (binary) cross-entropy

Used when examples can be classified as several of possible classification classes. Notably **used with `sigmoid` 
activation** function.

```math
L(t,y) = -\sum_{i}{t_i \log(y_i)} - \sum_{i}{(1-t_i)\log(1-y_i)}
```

```math
\frac{\partial L(t,y)}{\partial y_i} = \frac{y_i - t_i}{y_i (1 - y_i)}
```

### Optimizers

$\beta_1$ is normally set to 0.9; $\beta_2$ is normally set to 0.999; $\epsilon$ is normally set to `1e-8`.
Default learning rate for `momentum` and `demon momentum` is ~0.01, whilst for `adam`-based updates, is ~0.001.

#### Momentum

```math
v_t = \beta_1 v_{t-1} - \frac{\partial E}{\partial w}
```

```math
w_{t + 1} = w_t + \eta v_t
```

#### Adam

```math
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\cdot \frac{\partial E}{\partial w}
```

```math
v_t = \beta_2 v_{t-1} + (1 - \beta_2)\cdot \left(\frac{\partial E}{\partial w}\right)^2
```

```math
w_{t + 1} = w_t - \frac{\eta}{\sqrt{\frac{v_t}{1-\beta^t_2}} + \epsilon} \cdot \frac{m_t}{1-\beta^t_1}
```

Where $t$ increments with each time step. In my implementation, I increment $t$ for each batch update performed. 

#### Demon Momentum

```math
p_t = \frac{T - t}{T}
```

```math
\beta_t = \beta_1 \cdot \frac{p_t}{(1 - \beta_1) + \beta_1 p_t}
```

```math
v_t = \beta_t v_{t-1} - \frac{\partial E}{\partial w}
```

```math
w_{t + 1} = w_t + \eta v_t
```

for $T$ representing the total number of time steps in the cycle. In my implementation, I set $T$ to be the number of 
mini-batches in one epoch (iteration through entire training dataset). 

#### Demon Adam

```math
p_t = \frac{T - t}{T}
```

```math
\beta_t = \beta_1 \cdot \frac{p_t}{(1 - \beta_1) + \beta_1 p_t}
```

```math
m_t = \beta_t m_{t-1} + (1 - \beta_1)\cdot \frac{\partial E}{\partial w}
```

```math
v_t = \beta_2 v_{t-1} + (1 - \beta_2)\cdot \left(\frac{\partial E}{\partial w}\right)^2
```

```math
w_{t + 1} = w_t - \frac{\eta}{\sqrt{\frac{v_t}{1-\beta^t_2}} + \epsilon} \cdot \frac{m_t}{1-\beta^t_1}
```

### Activation functions

Let $x$ be $\text{net}$.

#### Sigmoid

```math
\varphi(x) = \frac{1}{1+e^{-x}}
```

```math
\varphi'(x) = \varphi(x)(1-\varphi(x)) 
```

#### Tanh

```math
\varphi(x) = \frac{e^x - e^{-x}}{e^x+e^{-x}}
```

```math
\varphi'(x) = 1 - \varphi(x)^2 
```

#### ReLU

```math
\varphi(x) = \max(0, x)
```

```math
\varphi'(x) = \begin{cases}
0 & \text{if } x < 0 \\
1 & \text{if } x > 0
\end{cases}
```

#### Leaky-ReLU

```math
\varphi(x) = \begin{cases}
0.01x & \text{if } x < 0 \\
x & \text{if } x > 0
\end{cases}
```

```math
\varphi'(x) = \begin{cases}
0.01 & \text{if } x < 0 \\
1 & \text{if } x > 0
\end{cases}
```

#### Softmax

```math
\varphi(x, i) = \frac{e^{x_i}}{\sum_{k}{e^{x_k}}}
```

Note, the implementation only considers its use with **categorical cross entropy**:

```math
\frac{\partial L(t, y)}{\partial y_i}\frac{d\varphi{(\text{net}_i)}}{d\text{net}_i} = y_i - t_i
```

### Note on GD, SGD, and mini-batch GD

Gradient descent performs updates after an epoch of weight updates is averaged, whereas mini-batch gradient descent does 
so with smaller batches, and stochastic gradient descent updates parameters after each training example.

### Parallelization

Simple parallelization is implemented through Java's `streams` API:

```java
IntStream.range(0, weights.length).parallel().forEach(i -> {
  // Perform operations
});

// Is equivalent to: 
for (int i = 0; i < weights.length; i++) {
  // Perform operations
}
```

### References

- [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)
- [Activation functions](https://en.wikipedia.org/wiki/Activation_function)
- [Optimizers](https://www.ruder.io/optimizing-gradient-descent/)
- [More on optimizers](https://johnchenresearch.github.io/demon/)
- [Demon (Decaying momentum)](https://akyrillidis.github.io/pubs/Conferences/Demon.pdf)
- Cross entropy and backpropagation: 
  [[1]](https://towardsdatascience.com/deriving-backpropagation-with-cross-entropy-loss-d24811edeaf9), 
  [[2]](http://neuralnetworksanddeeplearning.com/chap3.html#eqtn63)
- Based off Y8 me’s overcomplicated (and likely
  inaccurate) [code](https://github.com/JC-ProgJava/Building-Neural-Networks-From-Scratch).