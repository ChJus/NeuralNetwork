package lib;

public enum Optimizer {
  NONE, MOMENTUM, ADAM, DEMON_ADAM, DEMON_MOMENTUM
}

// https://www.ruder.io/optimizing-gradient-descent/
// https://johnchenresearch.github.io/demon/

// Optimizers loose benchmark (42,000 MNIST)
// 784-500-10
// Batch size: 50, Epochs: 5, Initialization: GAUSSIAN
// MOMENTUM       LR0.01:  65 seconds; 77.8, 89.5, 91.0, 92.1, 92.7
// DEMON_MOMENTUM LR0.01:  60 seconds; 77.4, 87.9, 90.0, 91.2, 91.8
// ADAM          LR0.001:  90 seconds; 67.6, 86.8, 89.4, 90.9, 91.7
// DADAM         LR0.001: 100 seconds; 63.4, 83.5, 87.1, 88.9, 90.0
// NONE           LR0.01:  65 seconds; 46.0, 72.3, 80.6, 84.3, 86.1

// CIFAR-10 benchmark
// 3072-1000-700-10
// Batch size: 100, Epochs: 3, Initialization: GAUSSIAN
// DEMON_ADAM       LR0.001: 15 minutes;  33.5, 38.9, 40.8
// MOMENTUM          LR0.01: 11 minutes;  23.1, 32.0, 34.7
// DEMON_MOMENTUM    LR0.01: 11 minutes;	21.6, 28.2, 31.5
// DEMON_ADAM        LR0.01: 14 minutes;  17.5, 24.9, 30.5
// ADAM              LR0.01: 15 minutes;  12.0, 21.4, 28.7
// NONE              LR0.01:  9 minutes;  11.9, 16.9, 20.2