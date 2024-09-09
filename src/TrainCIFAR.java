import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

public class TrainCIFAR {
  public static void main(String[] args) throws IOException, ClassNotFoundException {
    Network network = new Network(
        new int[]{3072, 256, 10},
        new ActivationFunction[]{ActivationFunction.LEAKY_RELU, ActivationFunction.SIGMOID},
        Initializer.GAUSSIAN);

    ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream("input-cifar10.ser"));
    double[][] input = (double[][]) objectInputStream.readObject();
    objectInputStream.close();

    objectInputStream = new ObjectInputStream(new FileInputStream("target-cifar10.ser"));
    double[][] target = (double[][]) objectInputStream.readObject();
    objectInputStream.close();

    for (int i = 0; i < 10; i++)
      network.train(input, target, 0.01, Error.MEAN_SQUARED, Optimizer.MOMENTUM, 50);

    // 3072-256-10, 10 epochs, LR 0.002, ADAM, Batch: 50; Train Acc.: _/_
  }
}