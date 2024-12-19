package lib;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

public class Layer implements Serializable {
  private static final Random random = new Random();
  public ActivationFunction activationFunction;
  public boolean isOutputLayer;

  double[] inputs;
  double[] weightedSumOutput; // before applying activation function

  double[][] weights;
  double[] deltaWeights;
  double[][] weightsAdjustments;

  double[] biases;
  double[] biasesAdjustments;

  transient double learningRate;

  public Layer(int in, int out, ActivationFunction activationFunction, boolean isOutputLayer, Initializer initializer) {
    this.weightedSumOutput = new double[out];
    this.isOutputLayer = isOutputLayer;
    this.activationFunction = activationFunction;

    this.weights = new double[in][out];
    this.weightsAdjustments = new double[in][out];
    this.deltaWeights = new double[out];

    this.biases = new double[out];
    this.biasesAdjustments = new double[out];

    for (int i = 0; i < in; i++) {
      for (int j = 0; j < out; j++) {
        weights[i][j] = switch (initializer) {
          case GAUSSIAN -> random.nextGaussian(0, 0.1);
          case RANDOM -> random.nextDouble() - 0.5;
          case ZERO -> 0;
        };
      }
    }

    for (int j = 0; j < out; j++) {
      biases[j] = switch (initializer) {
        case GAUSSIAN -> random.nextGaussian(0, 0.1);
        case RANDOM -> random.nextDouble() - 0.5;
        case ZERO -> 0;
      };
    }
  }

  public Layer(double[][] weights, double[] biases, boolean isOutputLayer, ActivationFunction activationFunction) {
    this.weightedSumOutput = new double[weights[0].length];
    this.isOutputLayer = isOutputLayer;
    this.activationFunction = activationFunction;

    this.weights = weights;
    this.weightsAdjustments = new double[weights.length][weights[0].length];
    this.deltaWeights = new double[weights[0].length];

    this.biases = biases;
    this.biasesAdjustments = new double[biases.length];
  }

  double[] feedforward(double[] inputs) {
    if (inputs.length != weights.length)
      throw new IllegalArgumentException("Input size " + inputs.length + " does not match weights size " + weights.length);

    this.inputs = inputs;

    double[] outputs = new double[weights[0].length];

    IntStream.range(0, weights[0].length).parallel().forEach(j -> {
      for (int i = 0; i < weights.length; i++) {
        outputs[j] += weights[i][j] * inputs[i];
      }
      outputs[j] += biases[j];
      weightedSumOutput[j] = outputs[j];

      if (activationFunction != ActivationFunction.SOFTMAX) {
        outputs[j] = activationFunction(outputs[j], false);
      }
    });

    if (activationFunction == ActivationFunction.SOFTMAX) {
      double max = Arrays.stream(outputs).max().getAsDouble();
      for (int i = 0; i < outputs.length; i++) {
        outputs[i] -= max;
        outputs[i] = Math.exp(outputs[i]);
      }
      double total = Arrays.stream(outputs).sum();
      for (int i = 0; i < outputs.length; i++) {
        outputs[i] /= total;
      }
    }

    /*
    for (int j = 0; j < weights[0].length; j++) {
      for (int i = 0; i < weights.length; i++) {
        outputs[j] += weights[i][j] * inputs[i];
      }
      outputs[j] += biases[j];
      this.weightedSumOutput[j] = outputs[j];
      outputs[j] = activationFunction(outputs[j], false);
    }
     */

    return outputs;
  }

  void learn(Layer nextLayer, double[] error, double learningRate, Optimizer optimizer, Error errorFunction) {
    this.learningRate = learningRate;
    Arrays.fill(deltaWeights, 0);

    if (error == null && isOutputLayer || !isOutputLayer && nextLayer == null)
      throw new IllegalArgumentException("Must have succeeding layer or error array to learn.");
    if (error != null && error.length != weights[0].length)
      throw new IllegalArgumentException("Mismatch between error array and output neurons array.");

    if (isOutputLayer) {
      for (int j = 0; j < weights[0].length; j++) {
        switch (errorFunction) {
          case MEAN_SQUARED:
            assert activationFunction != ActivationFunction.SOFTMAX;
            deltaWeights[j] = error[j] * activationFunction(weightedSumOutput[j], true);
            break;

          case MULTI_CLASS_CROSS_ENTROPY:
            assert activationFunction != ActivationFunction.SOFTMAX;
            if (activationFunction == ActivationFunction.SIGMOID) deltaWeights[j] = error[j];
            else deltaWeights[j] = error[j] * activationFunction(weightedSumOutput[j], true);
            break;

          case CATEGORICAL_CROSS_ENTROPY:
            if (activationFunction == ActivationFunction.SOFTMAX) deltaWeights[j] = error[j];
            else deltaWeights[j] = error[j] * activationFunction(weightedSumOutput[j], true);
            break;
        }
      }
    } else {
      IntStream.range(0, nextLayer.weights.length).parallel().forEach(j -> {
        for (int l = 0; l < nextLayer.weights[0].length; l++) {
          deltaWeights[j] += nextLayer.weights[j][l] * nextLayer.deltaWeights[l];
        }
        deltaWeights[j] *= activationFunction(weightedSumOutput[j], true);
      });
    }

    switch (optimizer) {
      case NONE:
        normal();
        break;
      case MOMENTUM:
        momentum();
        break;
      case DEMON_MOMENTUM:
        demonMomentum();
        break;
      case ADAM:
        adam();
        break;
      case DEMON_ADAM:
        demonAdam();
        break;
    }
  }

  void normal() {
    IntStream.range(0, weights.length).parallel().forEach(i -> {
      for (int j = 0; j < weights[i].length; j++) {
        weightsAdjustments[i][j] += deltaWeights[j] * inputs[i] * -learningRate;
      }
    });

    for (int j = 0; j < biases.length; j++) {
      biasesAdjustments[j] += deltaWeights[j] * -learningRate;
    }
  }

  void epochReset() {
    t = 0;
    b1 = 0.9;
    b2 = 0.999;
  }

  transient double[][] velocity;
  transient double[][] moment;
  transient double beta1 = 0.9;
  transient double beta2 = 0.999;
  transient double epsilon = 1e-8;
  transient double t = 0;
  transient double T;
  transient double b1 = beta1;
  transient double b2 = beta2;

  void momentum() {
    // gradient = deltaWeights[j] * inputs[i] (∆ * input)
    if (velocity == null) velocity = new double[weights.length + 1][weights[0].length];

    IntStream.range(0, weights.length).parallel().forEach(i -> {
      for (int j = 0; j < weights[i].length; j++) {
        velocity[i][j] = beta1 * velocity[i][j] - deltaWeights[j] * inputs[i];
        weightsAdjustments[i][j] += velocity[i][j] * learningRate;
      }
    });

    for (int j = 0; j < biases.length; j++) {
      velocity[velocity.length - 1][j] = beta1 * velocity[velocity.length - 1][j] - deltaWeights[j];
      biasesAdjustments[j] += velocity[velocity.length - 1][j] * learningRate;
    }
  }

  void demonMomentum() {
    // gradient = deltaWeights[j] * inputs[i] (∆ * input)
    if (velocity == null) velocity = new double[weights.length + 1][weights[0].length];

    double p_t = (T - t) / T;
    double betaT = beta1 * (p_t / (1.0 - beta1 + beta1 * p_t));

    IntStream.range(0, weights.length).parallel().forEach(i -> {
      for (int j = 0; j < weights[i].length; j++) {
        velocity[i][j] = betaT * velocity[i][j] - deltaWeights[j] * inputs[i];
        weightsAdjustments[i][j] += velocity[i][j] * learningRate;
      }
    });

    for (int j = 0; j < biases.length; j++) {
      velocity[velocity.length - 1][j] = betaT * velocity[velocity.length - 1][j] - deltaWeights[j];
      biasesAdjustments[j] += velocity[velocity.length - 1][j] * learningRate;
    }
  }

  void adam() {
    if (moment == null || velocity == null) {
      velocity = new double[weights.length + 1][weights[0].length];
      moment = new double[weights.length + 1][weights[0].length];
    }

    IntStream.range(0, weights.length).parallel().forEach(i -> {
      for (int j = 0; j < weights[i].length; j++) {
        moment[i][j] = beta1 * moment[i][j] + (1.0 - beta1) * deltaWeights[j] * inputs[i];
        velocity[i][j] = beta2 * velocity[i][j] + (1.0 - beta2) * Math.pow(deltaWeights[j] * inputs[i], 2.0);

        weightsAdjustments[i][j] += -learningRate / (Math.sqrt(velocity[i][j] / (1.0 - b2)) + epsilon) * (moment[i][j] / (1.0 - b1));
      }
    });

    for (int j = 0; j < biases.length; j++) {
      moment[moment.length - 1][j] = beta1 * moment[moment.length - 1][j] + (1.0 - beta1) * deltaWeights[j];
      velocity[velocity.length - 1][j] = beta2 * velocity[velocity.length - 1][j] + (1.0 - beta2) * Math.pow(deltaWeights[j], 2.0);

      biasesAdjustments[j] += -learningRate / (Math.sqrt(velocity[velocity.length - 1][j] / (1.0 - b2)) + epsilon) * (moment[moment.length - 1][j] / (1.0 - b1));
    }

    b1 *= beta1;
    b2 *= beta2;
  }

  void demonAdam() {
    if (moment == null || velocity == null) {
      velocity = new double[weights.length + 1][weights[0].length];
      moment = new double[weights.length + 1][weights[0].length];
    }

    double p_t = (T - t) / T;
    double betaT = beta1 * (p_t / (1.0 - beta1 + beta1 * p_t));

    IntStream.range(0, weights.length).parallel().forEach(i -> {
      for (int j = 0; j < weights[i].length; j++) {
        moment[i][j] = betaT * moment[i][j] + (1.0 - beta1) * deltaWeights[j] * inputs[i];
        velocity[i][j] = beta2 * velocity[i][j] + (1.0 - beta2) * Math.pow(deltaWeights[j] * inputs[i], 2.0);

        weightsAdjustments[i][j] += -learningRate / (Math.sqrt(velocity[i][j] / (1.0 - b2)) + epsilon) * (moment[i][j] / (1.0 - b1));
      }
    });

    for (int j = 0; j < biases.length; j++) {
      moment[moment.length - 1][j] = betaT * moment[moment.length - 1][j] + (1.0 - beta1) * deltaWeights[j];
      velocity[velocity.length - 1][j] = beta2 * velocity[velocity.length - 1][j] + (1.0 - beta2) * Math.pow(deltaWeights[j], 2.0);

      biasesAdjustments[j] += -learningRate / (Math.sqrt(velocity[velocity.length - 1][j] / (1.0 - b2)) + epsilon) * moment[moment.length - 1][j] / (1.0 - b1);
    }
    b1 *= beta1;
    b2 *= beta2;
  }

  double activationFunction(double x, boolean derivative) {
    switch (activationFunction) {
      case RELU:
        if (!derivative) return Math.max(x, 0);
        else return x < 0 ? 0 : 1;
      case LEAKY_RELU:
        if (!derivative) return x <= 0 ? 0.01 * x : x;
        else return x <= 0 ? 0.01 : 1;
      case SIGMOID:
        if (!derivative) return 1.0 / (1.0 + Math.exp(-x));
        else return (1.0 / (1.0 + Math.exp(-x))) * (1.0 - (1.0 / (1.0 + Math.exp(-x))));
      case TANH:
        if (!derivative) return (Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x));
        else return 1 - Math.pow((Math.exp(x) - Math.exp(-x)) / (Math.exp(x) + Math.exp(-x)), 2);
    }
    throw new RuntimeException(activationFunction + " is not a valid activation function.");
  }

  void updateWeights(double BATCH_SIZE) {
    for (int i = 0; i < weights.length; i++) {
      for (int j = 0; j < weights[i].length; j++) {
        weights[i][j] += weightsAdjustments[i][j] / BATCH_SIZE;
      }
      Arrays.fill(weightsAdjustments[i], 0);
    }

    for (int j = 0; j < biases.length; j++) {
      biases[j] += biasesAdjustments[j] / BATCH_SIZE;
    }
    Arrays.fill(biasesAdjustments, 0);
  }

  @Serial
  private void readObject(ObjectInputStream stream) throws IOException, ClassNotFoundException {
    stream.defaultReadObject(); // Deserialize the non-transient data

    // re-initialize transient fields to a well-defined value
    learningRate = 0.0;
    velocity = null;
    moment = null;
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;
    t = 0;
    T = 0;
    b1 = beta1;
    b2 = beta2;
  }
}