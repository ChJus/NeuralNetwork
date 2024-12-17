package main;

import lib.Network;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.HashMap;
import java.util.Map;

public class TestCIFAR {
  public static void main(String[] args) throws IOException, ClassNotFoundException {
    Map<Integer, String> labelNames = new HashMap<>();
    labelNames.put(0, "airplane");
    labelNames.put(1, "automobile");
    labelNames.put(2, "bird");
    labelNames.put(3, "cat");
    labelNames.put(4, "deer");
    labelNames.put(5, "dog");
    labelNames.put(6, "frog");
    labelNames.put(7, "horse");
    labelNames.put(8, "ship");
    labelNames.put(9, "truck");

    ObjectInputStream ois = new ObjectInputStream(new FileInputStream("cifar-10-network-temp.ser"));
    Network network = (Network) ois.readObject();
    ois.close();

    FileWriter fw = new FileWriter("out.csv");
    fw.write("id,label\n");

    for (int i = 1; i <= 300000; i++) {
      if (i % 1000 == 0) System.out.println(i + " / " + 300000);
      File file = new File("/Users/JC/Downloads/test/" + i + ".png");

      BufferedImage bufferedImage = ImageIO.read(file);
      double[] data = getPixels(bufferedImage);

      double[] result = network.feedforward(data);
      fw.write(i + "," + labelNames.get(network.getMax(result)) + "\n");
    }
    fw.close();
  }

  private static double[] getPixels(BufferedImage bufferedImage) {
    double[] vals = new double[bufferedImage.getWidth() * bufferedImage.getHeight() * 3];
    int[] pixel;
    int index = 0;
    for (int y = 0; y < bufferedImage.getHeight(); y++) {
      for (int x = 0; x < bufferedImage.getWidth(); x++) {
        pixel = bufferedImage.getRaster().getPixel(x, y, new int[3]);
        vals[index] = pixel[0] / 255.0;
        vals[index + 1] = pixel[1] / 255.0;
        vals[index + 2] = pixel[2] / 255.0;
        index += 3;
      }
    }
    return vals;
  }
}
