import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Main {
  public static void main(String[] args) throws FileNotFoundException {
    double[][] inputs = new double[1272][784];

    File file = new File("test.csv");
    Scanner in = new Scanner(file);
    in.nextLine();
    for (int i = 0; i < 1272; i++) {
      double[] vals = new double[784];
      String[] v = in.nextLine().trim().split(",");
      for (int j = 1; j < 785; j++) {
        vals[j - 1] = Double.parseDouble(v[j]);
      }
      inputs[i] = vals;
    }
  }
}