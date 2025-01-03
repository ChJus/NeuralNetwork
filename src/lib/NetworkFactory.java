package lib;

import java.io.*;
import java.util.Arrays;
import java.util.Scanner;

public final class NetworkFactory {
  private NetworkFactory() {

  }

  public static Network fromSerialization(String serializationFilepath) throws IOException, ClassNotFoundException {
    ObjectInputStream ois = new ObjectInputStream(new FileInputStream(serializationFilepath));
    Network network = (Network) ois.readObject();
    ois.close();
    return network;
  }

  public static Network fromTextFile(String textFilepath) throws IOException {
    Scanner in = new Scanner(new File(textFilepath));
    String[] build = in.nextLine().split(" ");
    String[] activations = in.nextLine().split(" ");

    Layer[] layers = new Layer[build.length - 1];

    for (int i = 0; i < layers.length; i++) {
      double[][] weights = new double[Integer.parseInt(build[i])][Integer.parseInt(build[i + 1])];
      double[] biases = new double[Integer.parseInt(build[i + 1])];

      String[] w = in.nextLine().replace("[", "").replace("]", "\n").replace(",", "").trim().split("\n");
      String[] b = in.nextLine().replace("[", "").replace("]", "").split(", ");

      for (int j = 0; j < w.length; j++) {
        String[] vals = w[j].trim().split(" ");
        for (int k = 0; k < vals.length; k++) {
          weights[j][k] = Double.parseDouble(vals[k]);
        }
      }

      for (int j = 0; j < b.length; j++) {
        biases[j] = Double.parseDouble(b[j]);
      }

      in.nextLine();

      layers[i] = new Layer(weights, biases, i == layers.length - 1, ActivationFunction.valueOf(activations[i]));
    }

    return new Network(layers);
  }

  public static void serialize(Network network, String destination) throws IOException {
    ObjectOutputStream objectOutputStream = new ObjectOutputStream(new FileOutputStream(destination));
    objectOutputStream.writeObject(network);
    objectOutputStream.close();
  }

  public static void export(Network network, String destination) throws IOException {
    FileWriter fw = new FileWriter(destination);

    for (int i = 0; i < network.layers.length; i++) {
      fw.write(network.layers[i].weights.length + (i == network.layers.length - 1 ? (" " + network.layers[i].weights[0].length + "\n") : " "));
    }

    for (int i = 0; i < network.layers.length; i++) {
      fw.write(network.layers[i].activationFunction + (i == network.layers.length - 1 ? "\n" : " "));
    }

    for (Layer l : network.layers) {
      fw.write(Arrays.deepToString(l.weights) + "\n");
      fw.write(Arrays.toString(l.biases) + "\n\n");
    }

    fw.close();
  }
}
