import java.io.*;
import java.util.StringTokenizer;

public class main {

	static int width = 28, height = 28;
	static int size = width * height + 1;
	static int nTrain = 49000, nTest = 21000;
	static int epochs = 5;
	static int loss = 0;
	static MLP mlp = new MLP(); // 3 layers
	static int image[][] = new int[width][height];
	static double input[] = new double[size], output[] = new double[10];
	static double lossRate = 0;

	static BufferedReader br;
	static BufferedWriter bw;

	private static void readImage() throws IOException {
		input[size - 1] = 1.0;
		StringTokenizer st;
		for (int i = 0; i < height; ++i) {
			st = new StringTokenizer(br.readLine());
			for (int j = 0; j < width; ++j)
				image[i][j] = Integer.parseInt(st.nextToken());
		}
		for (int i = 0; i < height; ++i)
			for (int j = 0; j < width; ++j)
				if (image[i][j] > 0) input[i * height + j] = 1.0;
				else input[i * height + j] = 0.0;
	}

	private static void setTarget(int label) {
		for (int i = 0; i < output.length; ++i)
			output[i] = 0.0;
		output[label] = 1.0;
	}

	public static void main(String args[]) throws IOException {
		bw = new BufferedWriter(new OutputStreamWriter(System.out));
		bw.write("Training...  |                    |");
		// Training data
		int progress = 0;
		for (int e = 1; e <= epochs; e++) {
			progress = nTrain / 10;
			bw.write("\nProgress Bar |");
			bw.flush();
			br = new BufferedReader(new InputStreamReader(new FileInputStream("MNIST.txt")));
			loss = 0;

			/* Training data learning */
			for (int i = 0; i < nTrain; ++i) {
				int label = Integer.parseInt(br.readLine());
				readImage();
				setTarget(label);
				mlp.learning(input, output);

				if ((i + 1) % progress == 0) { // percentage of completion
					progress += nTrain / 10;
					bw.write("#");
					bw.flush();
				}
			}
			progress = nTrain / 10;
			br = new BufferedReader(new InputStreamReader(new FileInputStream("MNIST.txt")));

			/* Training data predict */
			for (int i = 0; i < nTrain; ++i) {
				int label = Integer.parseInt(br.readLine());
				readImage();
				boolean prediction = (label == mlp.predict(input, output)) ? true : false;
				if (!prediction)
					++loss;
				if ((i + 1) % (progress) == 0) {
					progress += nTrain / 10;
					bw.write("#");
					if (i + 1 == nTrain) bw.write("|");
					bw.flush();
				}
			}
			lossRate = (double) (loss) / nTrain * 100.0;
			bw.write("\nEpochs: " + e + " Number of losses in Training data: " + loss + " / " + nTrain + "\n");
			bw.write("Training data loss rate: " + lossRate + "\n");
			bw.flush();
		}

		// test data
		bw.write("\nTest\nProgress Bar |");
		bw.flush();
		progress = nTest / 20;
		loss = 0;

		/* model test */
		for (int i = 0; i < nTest; ++i) {
			int label = Integer.parseInt(br.readLine());
			readImage();
			boolean prediction = (label == mlp.predict(input, output)) ? true : false;
			if (!prediction)
				++loss;
			if ((i + 1) % progress == 0) {
				progress += nTest / 20;
				bw.write("#");
				if (i + 1 == nTest) bw.write("|");
				bw.flush();
			}
		}
		lossRate = (double) (loss) / nTest * 100.0;
		bw.write("\nNumber of losses in test data: " + loss + " / " + nTest + "\n");
		bw.write("Test data loss rate: " + lossRate + "\n");
		bw.flush();
		br.close();
		bw.close();
	}
}
