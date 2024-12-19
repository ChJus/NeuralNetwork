package lib;

import java.io.Serializable;

public class Network implements Serializable {
  public final Layer[] layers;

  public Network(int[] layerSizes, ActivationFunction[] activationFunctions, Initializer initializer) {
    if (layerSizes.length != activationFunctions.length + 1)
      throw new IllegalArgumentException("Neurons array should be 1 more than activation function array.");

    layers = new Layer[layerSizes.length - 1];
    for (int i = 0; i < layerSizes.length - 1; i++) {
      layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], activationFunctions[i], i == layers.length - 1, initializer);
    }
  }

  public Network(Layer[] layers) {
    this.layers = layers;
  }

  public double[] feedforward(double[] inputs) {
    for (Layer l : layers) {
      inputs = l.feedforward(inputs);
    }
    return inputs;
  }

  public int getMax(double[] arr) {
    int index = 0;
    double value = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < arr.length; i++) {
      if (arr[i] > value) {
        index = i;
        value = arr[i];
      }
    }
    return index;
  }

  public void train(double[][] inputs, double[][] targets, double learningRate, Error error, Optimizer optimizer, int BATCH_SIZE) {
    if (targets.length != inputs.length)
      throw new IllegalArgumentException("Input and target arrays have mismatched size");

    // Set total iteration count T, used for demon adam
    for (Layer l : layers) {
      l.T = (double) inputs.length / BATCH_SIZE;
    }

    int counter = 0;
    int correct = 0;
    double printError = 0;
    double[][] guesses = new double[targets[0].length][targets[0].length];

    long start = System.currentTimeMillis();
    for (int i = 0; i < inputs.length; i++) {
      counter++;
      double[] result = feedforward(inputs[i]);

      if (getMax(result) == getMax(targets[i])) {
        correct++;
      }
      guesses[getMax(result)][getMax(targets[i])]++;

      double[] errorArray = new double[result.length];

      for (int j = 0; j < result.length; j++) {
        switch (error) {
          case MEAN_SQUARED:
            errorArray[j] = result[j] - targets[i][j];
            printError += Math.pow((result[j] - targets[i][j]), 2) * 0.5;
            break;

          case MULTI_CLASS_CROSS_ENTROPY:
            if (layers[layers.length - 1].activationFunction == ActivationFunction.SIGMOID)
              errorArray[j] = result[j] - targets[i][j];
            else
              errorArray[j] = (result[j] - targets[i][j]) / (result[j] * (1 - result[j]));

            printError -= targets[i][j] * Math.log(result[j] + 1e-20) + (1 - targets[i][j]) * Math.log(1 - result[j] + 1e-20);
            break;

          case CATEGORICAL_CROSS_ENTROPY:
            if (layers[layers.length - 1].activationFunction == ActivationFunction.SOFTMAX)
              errorArray[j] = result[j] - targets[i][j];
            else
              errorArray[j] = -targets[i][j] / result[j];

            printError -= targets[i][j] * Math.log(result[j] + 1e-20);
            break;
        }
      }

      layers[layers.length - 1].learn(null, errorArray, learningRate, optimizer, error);
      for (int l = layers.length - 2; l >= 0; l--) {
        layers[l].learn(layers[l + 1], null, learningRate, optimizer, null);
      }

      if (counter % BATCH_SIZE == 0) {
        for (int l = layers.length - 1; l >= 0; l--) {
          layers[l].updateWeights(BATCH_SIZE);
          layers[l].t++;
        }
      }
      if (counter % 1000 == 0) {
        String str = String.format("%-60.60s",
            "\rTime: " + ((System.currentTimeMillis() - start) / 1000.0) + " seconds.") + " |" + String.format("%-10.10s", ("=").repeat((int) ((double) counter / (double) inputs.length * 10.0))) + "|";
        System.out.print(str);
      }
    }

    for (Layer l : layers) {
      l.epochReset();
    }

    String str = "\rEpoch time: " + ((System.currentTimeMillis() - start) / 1000.0) + " seconds.";
    System.out.print(str);
    System.out.println();

    System.out.println("Error: " + String.format("%.6f", printError) + "\t" +
        String.format(("%" + (int) Math.ceil(Math.log10(inputs.length)) + "d"), correct) + "/" + inputs.length + "\t" +
        String.format("%.2f", 100.0 * correct / inputs.length) + "%");

    for (int index = 0; index < guesses.length; index++) {
      System.out.print("[");
      for (int indice = 0; indice < guesses[index].length; indice++) {
        if (index == indice) {
          System.out.print("\u001B[32m" + guesses[index][indice] + "\u001B[0m");
        } else {
          System.out.print(guesses[index][indice]);
        }

        System.out.print((indice == guesses[index].length - 1) ? "]" : ", ");
      }
      System.out.println();
    }
    System.out.println();
  }
}
