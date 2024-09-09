class Network {
  Layer[] layers;
  double[] output;
  double printError;

  Network(int[] layerSizes, ActivationFunction[] activationFunctions, Initializer initializer) {
    if (layerSizes.length != activationFunctions.length + 1)
      throw new IllegalArgumentException("Neurons array should be 1 more than activation function array.");

    this.output = new double[layerSizes[layerSizes.length - 1]];

    layers = new Layer[layerSizes.length - 1];
    for (int i = 0; i < layerSizes.length - 1; i++) {
      layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], activationFunctions[i], i == layers.length - 1, initializer);
    }
  }

  double[] feedforward(double[] inputs) {
    for (Layer l : layers) {
      inputs = l.feedforward(inputs);
    }
    return inputs;
  }

  int getMax(double[] arr) {
    int index = 0;
    double value = Double.MAX_VALUE;
    for (int i = 0; i < arr.length; i++) {
      if (arr[i] < value) {
        index = i;
        value = arr[i];
      }
    }
    return index;
  }

  void train(double[][] inputs, double[][] targets, double learningRate, Error error, Optimizer optimizer, int BATCH_SIZE) {
    printError = 0;
    if (targets.length != inputs.length)
      throw new IllegalArgumentException("Input and target arrays have mismatched size");

    int counter = 0;
    int correct = 0;
    for (int i = 0; i < inputs.length; i++) {
      counter++;
      double[] result = feedforward(inputs[i]);

      if (getMax(result) == getMax(targets[i])) correct++;

      double[] errorArray = new double[result.length];

      for (int j = 0; j < result.length; j++) {
        switch (error) {
          case MEAN_SQUARED:
            errorArray[j] = result[j] - targets[i][j];
            printError += Math.pow((result[j] - targets[i][j]), 2) * 0.5;
            break;
        }
      }
      layers[layers.length - 1].learn(null, errorArray, learningRate, optimizer);
      if (counter % BATCH_SIZE == 0) {
        layers[layers.length - 1].updateWeights(BATCH_SIZE);
      }
      for (int l = layers.length - 2; l >= 0; l--) {
        layers[l].learn(layers[l + 1], null, learningRate, optimizer);
        if (counter % BATCH_SIZE == 0) {
          layers[l].updateWeights(BATCH_SIZE);
        }
      }
    }
    System.out.println(printError + " " + correct + "/" + inputs.length);
  }
}
