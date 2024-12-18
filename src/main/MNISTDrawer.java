package main;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;

public class MNISTDrawer {
  public static void main(String[] args) throws IOException, ClassNotFoundException {
    double[][] input = new double[28000][784];

    File file = new File("testMNIST.csv");
    Scanner in = new Scanner(file);
    in.nextLine();
    for (int i = 0; i < 28000; i++) {
      double[] vals = new double[784];
      String[] v = in.nextLine().trim().split(",");
      for (int j = 0; j < 784; j++) {
        vals[j] = Double.parseDouble(v[j]);
      }
      input[i] = vals;
    }

    for (int i = 1; i <= 300; i++) {
      int imageN = new Random().nextInt(28000);
      BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
      int[] pixels = new int[784];
      for (int j = 0; j < 784; j++) {
        pixels[j] = (int) (input[imageN][j]);
      }
      image.getRaster().setPixels(0, 0, 28, 28, pixels);
      ImageIO.write(image, "png", new File("docs/mnist-examples/" + i + ".png"));
    }
  }
}
