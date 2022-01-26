class Layer {
	int num;
	double out[], d[];
	double weight[][], delta[][];
};

public class MLP {
	private int inputLayerNode = 28 * 28 + 1;
	private int hiddenLayerNode = 128;
	private int outputLayerNode = 10;
	private int iLayer, hLayer, oLayer;
	private int layerNum;
	private double target[];
	private double learningRate = 0.1;
	private double alpha = 0.9; // momentum
	Layer layer[];

	public MLP() { // Settings
		layerNum = 3; // input - hidden - output
		iLayer = 0; // input layer step
		oLayer = layerNum - 1; // ouput layer step
		hLayer = oLayer - 1; // hidden layer step
		layer = new Layer[layerNum];
		for (int i = 0; i < layerNum; i++)
			layer[i] = new Layer();
		layer[iLayer].num = inputLayerNode;
		layer[hLayer].num = hiddenLayerNode;
		layer[oLayer].num = outputLayerNode;
		target = new double[layer[oLayer].num]; // actual output value
		for (int i = 0; i < layerNum; i++) {
			if (i != iLayer)
				layer[i].d = new double[layer[i].num];
			layer[i].out = new double[layer[i].num];
			if (i != oLayer) {
				layer[i].weight = new double[layer[i].num][layer[i + 1].num];
				layer[i].delta = new double[layer[i].num][layer[i + 1].num];
			}
		}
		
		// given as random real numbers with uniform distribution
		for (int i = 0; i < oLayer; i++)
			for (int j = 0; j < layer[i].num; j++)
				for (int k = 0; k < layer[i + 1].num; k++) {
					layer[i].weight[j][k] = (double) (Math.random() * (4.8 / layer[i].weight.length))
							- (2.4 / layer[i].weight.length);
					int t = (int) (Math.random() * 2) + 1;
					if (t == 1) layer[i].weight[j][k] *= -1.0;
				}
	}

	public void learning(double Input[], double Output[]) {
		for (int i = 0; i < layer[iLayer].num; i++)
			layer[0].out[i] = Input[i];
		for (int i = 0; i < layer[oLayer].num; i++)
			target[i] = Output[i];
		for (int i = 0; i < oLayer; i++)
			for (int j = 0; j < layer[i].num; j++)
				for (int k = 0; k < layer[i + 1].num; k++)
					layer[i].delta[j][k] = 0.0;
		perceptron();
		backPropagation();
	}

	private void perceptron() {
		for (int i = 1; i < layerNum; i++)
			for (int j = 0; j < layer[i].num; j++) {
				double x = 0.0;
				for (int k = 0; k < layer[i - 1].num; k++)
					x += layer[i - 1].out[k] * layer[i - 1].weight[k][j];
				layer[i].out[j] = sigmoid(x); // compute output
			}
	}

	private double sigmoid(double x) {
		return 1.0 / (1.0 + Math.exp(-x));
	}

	private void backPropagation() {
		gradient();
		for (int i = 0; i < oLayer; i++)
			for (int j = 0; j < layer[i].num; j++)
				for (int k = 0; k < layer[i + 1].num; k++) {
					double delta = layer[i].delta[j][k];
					double out = layer[i].out[j];
					double d = layer[i + 1].d[k];
					layer[i].delta[j][k] = learningRate * d * out + alpha * delta;
					layer[i].weight[j][k] += layer[i].delta[j][k];
				}
	}

	private void gradient() { // error gradient
		for (int i = 0; i < layer[oLayer].num; i++) {
			double value = layer[oLayer].out[i];
			layer[oLayer].d[i] = value * (1 - value) * (target[i] - value); // Output error gradient
		}
		for (int hLayer = 1; hLayer < oLayer; hLayer++)
			for (int j = 0; j < layer[hLayer].num; j++) {
				double sum = 0.0;
				for (int k = 0; k < layer[oLayer].num; k++)
					sum += layer[oLayer].d[k] * layer[hLayer].weight[j][k];
				double value = layer[hLayer].out[j];
				layer[hLayer].d[j] = sum * value * (1 - value); // hidden error gradient
			}
	}

	public int predict(double input[], double output[]) {
		for (int i = 0; i < layer[iLayer].num; i++)
			layer[iLayer].out[i] = input[i];
		perceptron();
		for (int i = 0; i < layer[oLayer].num; i++)
			output[i] = layer[oLayer].out[i];
		int predictedLabel = 0;
		for (int i = 1; i < output.length; ++i)
			if (output[predictedLabel] < output[i])
				predictedLabel = i;
		return predictedLabel;
	}
}
