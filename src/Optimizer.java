enum Optimizer {
  NONE, MOMENTUM, ADAM, NADAM
}
// https://www.ruder.io/optimizing-gradient-descent/
// Optimizers loose benchmark (42,000 MNIST)
// LR: 0.002, Batch size: 50, Epochs: 10, Initialization: GAUSSIAN
// MOMENTUM:  37276/42000
// ADAM:      37211/42000
// NADAM:     36853/42000
// NONE:      35206/42000