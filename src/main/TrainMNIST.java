package main;

import lib.Error;
import lib.*;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;

public class TrainMNIST {
  public static void main(String[] args) throws IOException {
//    Network network = NetworkFactory.createFrom("mnist-network.txt");
    Network network = new Network(
        new int[]{784, 500, 10},
        new ActivationFunction[]{ActivationFunction.SIGMOID, ActivationFunction.SIGMOID},
        Initializer.GAUSSIAN);

    double[][] inputs = new double[42000][784];
    double[][] targets = new double[42000][10];

    File file = new File("trainMNIST.csv");
    Scanner in = new Scanner(file);
    in.nextLine();
    for (int i = 0; i < 42000; i++) {
      double[] vals = new double[784];
      String[] v = in.nextLine().trim().split(",");
      for (int j = 1; j < 785; j++) {
        vals[j - 1] = Double.parseDouble(v[j]);
      }
      targets[i][(int) (Double.parseDouble(v[0]))] = 1;
      inputs[i] = vals;
    }

    for (int i = 0; i < 5; i++) {
      network.train(inputs, targets, 0.01, Error.MEAN_SQUARED, Optimizer.DEMON_MOMENTUM, 50);
      NetworkFactory.serialize(network, "mnist-network-temp.ser");
    }

    NetworkFactory.serialize(network, "mnist-network.ser");
    NetworkFactory.export(network, "mnist-network.txt");
  }
}
