import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

class Layer implements Serializable {
  static Random random = new Random();
  Initializer initializer;
  ActivationFunction activationFunction;
  boolean isOutputLayer;
  double[] inputs;
  double[] weightedSumOutput; // before applying activation function

  double[][] weights;
  double[] deltaWeights;
  double[][] weightsAdjustments;

  double[] biases;
  double[] biasesAdjustments;

  double learningRate;

  public Layer(int in, int out, ActivationFunction activationFunction, boolean isOutputLayer, Initializer initializer) {
    this.weightedSumOutput = new double[out];
    this.isOutputLayer = isOutputLayer;
    this.initializer = initializer;
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
        case GAUSSIAN -> random.nextGaussian();
        case RANDOM -> random.nextDouble() - 0.5;
        case ZERO -> 0;
      };
    }
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
      this.weightedSumOutput[j] = outputs[j];
      outputs[j] = activationFunction(outputs[j], false);
    });

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

  void resetCache() {
    Arrays.fill(deltaWeights, 0);

    // TODO: may need to call this only after one entire epoch (i.e., after Network.train() finishes)
    //       currently set to reset after every mini-batch as results appear to be better.
    resetOptimizerCache();
  }

  void resetOptimizerCache() {
    oldWeightsAdjustments = null;
    oldBiasesAdjustments = null;
    beta1 *= beta1;
    beta2 *= beta2;
    oldWeightsAdjustments = null;
    oldBiasesAdjustments = null;
    v = null;
    v_hat = null;
    m = null;
    m_hat = null;
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;
  }

  void learn(Layer nextLayer, Error lossType, double[] error, double learningRate, Optimizer optimizer) {
    resetCache();
    this.learningRate = learningRate;
    if (error == null && isOutputLayer || !isOutputLayer && nextLayer == null)
      throw new IllegalArgumentException("Must have succeeding layer or error array to learn.");
    if (error != null && error.length != weights[0].length)
      throw new IllegalArgumentException("Mismatch between error array and output neurons array.");

    if (isOutputLayer) {
      for (int j = 0; j < weights[0].length; j++) {
        deltaWeights[j] = switch (lossType) {
          case Error.MEAN_SQUARED -> error[j] * activationFunction(weightedSumOutput[j], true);
          case Error.CROSS_ENTROPY -> error[j];
        };
      }
    } else {
      for (int j = 0; j < nextLayer.weights.length; j++) {
        for (int l = 0; l < nextLayer.weights[0].length; l++) {
          deltaWeights[j] += nextLayer.weights[j][l] * nextLayer.deltaWeights[l];
        }
        deltaWeights[j] *= activationFunction(weightedSumOutput[j], true);
      }
    }

    switch (optimizer) {
      case NONE -> learnNormal();
      case ADAM -> learnAdam();
      case NADAM -> learnNadam();
      case MOMENTUM -> learnMomentum();
    }
  }

  void learnNormal() {
    for (int i = 0; i < weights.length; i++) {
      for (int j = 0; j < weights[i].length; j++) {
        // error is assumed to be given by network and represents derivative of error function values
        weightsAdjustments[i][j] += deltaWeights[j] * inputs[i];
      }
    }

    for (int j = 0; j < biases.length; j++) {
      biasesAdjustments[j] += deltaWeights[j];
    }
  }

  double[][] oldWeightsAdjustments = null;
  double[] oldBiasesAdjustments = null;

  void learnMomentum() {
    if (oldWeightsAdjustments == null || oldBiasesAdjustments == null) {
      oldWeightsAdjustments = new double[weights.length][weights[0].length];
      oldBiasesAdjustments = new double[biases.length];
    }

    double mu = 0.9;

    for (int i = 0; i < weights.length; i++) {
      for (int j = 0; j < weights[i].length; j++) {
        // error is assumed to be given by network and represents derivative of error function values
        weightsAdjustments[i][j] += deltaWeights[j] * inputs[i];
        weightsAdjustments[i][j] += oldWeightsAdjustments[i][j] * mu;
      }
    }

    for (int j = 0; j < biases.length; j++) {
      biasesAdjustments[j] += deltaWeights[j];
      biasesAdjustments[j] += oldBiasesAdjustments[j] * mu;
    }

    oldWeightsAdjustments = weightsAdjustments.clone();
    oldBiasesAdjustments = biasesAdjustments.clone();
  }

  double[][] m = null, m_hat = null;
  double[][] v = null, v_hat = null;
  double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;

  void learnAdam() {
    if (m == null && v == null) {
      m = new double[weights.length][weights[0].length];
      v = new double[weights.length][weights[0].length];
      m_hat = new double[weights.length][weights[0].length];
      v_hat = new double[weights.length][weights[0].length];
    }

    for (int i = 0; i < weights.length; i++) {
      for (int j = 0; j < weights[i].length; j++) {
        // error is assumed to be given by network and represents derivative of error function values
        weightsAdjustments[i][j] += deltaWeights[j] * inputs[i];
        m[i][j] = beta1 * m[i][j] + (1 - beta1) * weightsAdjustments[i][j];
        v[i][j] = beta2 * v[i][j] + (1 - beta2) * weightsAdjustments[i][j] * weightsAdjustments[i][j];
        m_hat[i][j] = m[i][j] / (1 - beta1);
        v_hat[i][j] = v[i][j] / (1 - beta2);
        weightsAdjustments[i][j] = m_hat[i][j] / (Math.sqrt(v_hat[i][j]) + epsilon);
      }
    }

    for (int j = 0; j < biases.length; j++) {
      biasesAdjustments[j] += deltaWeights[j];
    }
  }

  void learnNadam() {
    double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;

    if (m == null && v == null) {
      m = new double[weights.length][weights[0].length];
      v = new double[weights.length][weights[0].length];
      m_hat = new double[weights.length][weights[0].length];
      v_hat = new double[weights.length][weights[0].length];
    }

    for (int i = 0; i < weights.length; i++) {
      for (int j = 0; j < weights[i].length; j++) {
        // error is assumed to be given by network and represents derivative of error function values
        weightsAdjustments[i][j] += deltaWeights[j] * inputs[i];
        m[i][j] = beta1 * m[i][j] + (1 - beta1) * weightsAdjustments[i][j];
        v[i][j] = beta2 * v[i][j] + (1 - beta2) * weightsAdjustments[i][j] * weightsAdjustments[i][j];
        m_hat[i][j] = m[i][j] / (1 - beta1);
        v_hat[i][j] = v[i][j] / (1 - beta2);
        weightsAdjustments[i][j] = (0.9 * m_hat[i][j] + (1 - 0.9) * weightsAdjustments[i][j] / (1 - beta1)) / (Math.sqrt(v_hat[i][j]) + epsilon);
      }
    }

    for (int j = 0; j < biases.length; j++) {
      biasesAdjustments[j] += deltaWeights[j];
    }
  }

  double[] softmaxCache = null;

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
      case SOFTMAX:
        if (!derivative) {

        } else {

        } // todo: implement softmax
    }
    throw new RuntimeException(activationFunction + " is not a valid activation function.");
  }

  void updateWeights(double BATCH_SIZE) {
    for (int i = 0; i < weights.length; i++) {
      for (int j = 0; j < weights[i].length; j++) {
        weights[i][j] -= learningRate * weightsAdjustments[i][j] / BATCH_SIZE;
      }
      Arrays.fill(weightsAdjustments[i], 0);
    }

    for (int j = 0; j < biases.length; j++) {
      biases[j] -= learningRate * biasesAdjustments[j] / BATCH_SIZE;
    }
    Arrays.fill(biasesAdjustments, 0);
  }
}