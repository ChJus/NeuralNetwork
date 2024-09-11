import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

public class TrainCIFAR {
  public static void main(String[] args) throws IOException, ClassNotFoundException {
    Network network = new Network(
        new int[]{3072, 800, 400, 10},
        new ActivationFunction[]{ActivationFunction.LEAKY_RELU, ActivationFunction.TANH, ActivationFunction.SIGMOID},
        Initializer.GAUSSIAN);

    ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream("input-cifar10.ser"));
    double[][] input = (double[][]) objectInputStream.readObject();
    objectInputStream.close();

    objectInputStream = new ObjectInputStream(new FileInputStream("target-cifar10.ser"));
    double[][] target = (double[][]) objectInputStream.readObject();
    objectInputStream.close();

    for (int i = 0; i < 20; i++)
      network.train(input, target, 0.005, Error.MEAN_SQUARED, Optimizer.ADAM, 20);

    // should theoretically be able to reach ~50%
    // 3072-256-10 (LEAKY_RELU, SIGMOID), 10 epochs, LR 0.01, NONE, Batch: 10; Train Acc.: 15250/50000
    // 3072-512-256-10 (SIGMOID, SIGMOID, SIGMOID), 10 epochs, LR 0.007, NONE, Batch: 10; Train Acc.:
  }
}