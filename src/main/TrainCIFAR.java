package main;

import lib.Error;
import lib.Network;
import lib.NetworkFactory;
import lib.Optimizer;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

public class TrainCIFAR {
  public static void main(String[] args) throws IOException, ClassNotFoundException {
//    Network network = new Network(
//        new int[]{3072, 1400, 1000, 500, 10},
//        new lib.ActivationFunction[]{lib.ActivationFunction.SIGMOID, lib.ActivationFunction.SIGMOID, lib.ActivationFunction.SIGMOID, lib.ActivationFunction.SIGMOID},
//        lib.Initializer.GAUSSIAN);

    Network network = NetworkFactory.fromSerialization("cifar-10-network-temp.ser");
//    Network network = NetworkFactory.createFrom("cifar-10-network.txt");

    ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream("input-cifar10.ser"));
    double[][] input = (double[][]) objectInputStream.readObject();
    objectInputStream.close();

    objectInputStream = new ObjectInputStream(new FileInputStream("target-cifar10.ser"));
    double[][] target = (double[][]) objectInputStream.readObject();
    objectInputStream.close();

    for (int i = 0; i < 5; i++) {
      network.train(input, target, 0.001, Error.MEAN_SQUARED, Optimizer.MOMENTUM, 100);
      NetworkFactory.serialize(network, "cifar-10-network-temp.ser");
    }

    NetworkFactory.serialize(network, "cifar-10-network.ser");
    NetworkFactory.export(network, "cifar-10-network.txt");
  }
}