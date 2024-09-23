import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.IntStream;

public class MLPChurnAnalysis {

    public static void main(String[] args) {
        List<String[]> data = loadDataSet("/Users/bastionii/Documents/GitHub/Machine Learning/data/bank_data.csv");
        double[][] features = prepareData(data);

        // dataset is not normalized right now

        double[][] normalizedFeatures = normalizeFeatures(features);

        // Define the number of inputs and outputs
        int numInputs = 10; // Number of features
        int numOutputs = 2; // Binary classification (0 or 1)

        MultiLayerNetwork model = buildModel(numInputs, numOutputs);

        // Initialize the model
        model.init();
        model.setListeners(new ScoreIterationListener(10));  // Print score every 10 iterations


        double trainFraction = 0.8; // 80% for training
        int[] target = extractTarget(data); // Extract target labels
        DataSet[] dataSets = splitDataIntoDataSet(normalizedFeatures, target, trainFraction);


        trainModel(model, dataSets[0], 100);  // Train the model with 100 epochs
        evaluateModel(model, dataSets[1]);

    }

    // Method to train the model
    private static void trainModel(MultiLayerNetwork model, DataSet trainingData, int numEpochs) {
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainingData);
        }
    }

    // Method to evaluate the model
    private static void evaluateModel(MultiLayerNetwork model, DataSet testData) {
        INDArray output = model.output(testData.getFeatures());
        Evaluation eval = new Evaluation(2); // binary classification
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats());
    }

    // Modify the splitDataIntoDataSet method

    private static DataSet[] splitDataIntoDataSet(double[][] features, int[] target, double trainFraction) {
        int trainSize = (int) (features.length * trainFraction);
        int testSize = features.length - trainSize;

        double[][] X_train = new double[trainSize][];
        int[] y_train = new int[trainSize];
        double[][] X_test = new double[testSize][];
        int[] y_test = new int[testSize];

        // Shuffle and split data
        Integer[] indices = IntStream.range(0, features.length).boxed().toArray(Integer[]::new);
        Collections.shuffle(Arrays.asList(indices), new Random());

        for (int i = 0; i < trainSize; i++) {
            X_train[i] = features[indices[i]];
            y_train[i] = target[indices[i]];
        }
        for (int i = 0; i < testSize; i++) {
            X_test[i] = features[indices[trainSize + i]];
            y_test[i] = target[indices[trainSize + i]];
        }

        // Convert to INDArrays
        INDArray trainFeatures = Nd4j.create(X_train);

        // One-hot encode labels for 2 classes (binary classification)
        INDArray trainLabels = Nd4j.create(trainSize, 2);  // 2 classes
        for (int i = 0; i < trainSize; i++) {
            trainLabels.putScalar(new int[] {i, y_train[i]}, 1.0);
        }

        INDArray testFeatures = Nd4j.create(X_test);
        INDArray testLabels = Nd4j.create(testSize, 2);  // 2 classes
        for (int i = 0; i < testSize; i++) {
            testLabels.putScalar(new int[] {i, y_test[i]}, 1.0);
        }

        // Convert to DataSets
        DataSet trainData = new DataSet(trainFeatures, trainLabels);
        DataSet testData = new DataSet(testFeatures, testLabels);

        return new DataSet[] {trainData, testData};


        /* splitData(double[][] features, int[] target, double trainFraction)
        Purpose: This function divides your dataset into training and testing sets.

        How it works:

        Input Parameters:
        features: A 2D array where each row represents a data point (e.g., customer attributes).
        target: An array of labels corresponding to the features (e.g., whether a customer exited or not).
        trainFraction: A value between 0 and 1 that specifies the proportion of the dataset to use for training
        (e.g., 0.8 means 80% for training, 20% for testing).
        Steps:

        Calculate Sizes:

        trainSize: The number of examples for the training set (based on the trainFraction).
                testSize: The remaining examples go to the testing set.
        Create Arrays:

        X_train, y_train: Arrays for training features and labels.
        X_test, y_test: Arrays for testing features and labels.
        Shuffle Data:

        An array of indices is created to randomize the order of the data points, which helps to ensure that the
        training and testing sets are representative of the overall dataset.
                Split the Data:

        The function populates the training and testing arrays by iterating through the shuffled indices.
        Output:

        It prints the sizes of the training and testing sets for verification. */


    }




    private static List<String[]> loadDataSet(String filePath) {
        List<String[]> data = new ArrayList<>();
        try (CSVReader reader = new CSVReader(new FileReader(filePath))) {
            data = reader.readAll();
        } catch (IOException | CsvException e) {
            e.printStackTrace();
        }
        return data;
    }

    private static double[][] prepareData(List<String[]> data) {
        List<double[]> featuresList = new ArrayList<>();
        List<Integer> targetList = new ArrayList<>();

        // Maps for Label Encoding
        Map<String, Integer> geographyMap = new HashMap<>();
        geographyMap.put("France", 0);
        geographyMap.put("Spain", 1);
        geographyMap.put("Germany", 2);

        Map<String, Integer> genderMap = new HashMap<>();
        genderMap.put("Female", 0);
        genderMap.put("Male", 1);

        for (int i = 1; i < data.size(); i++) {
            String[] row = data.get(i);

            double[] features = new double[10];

            try {
                features[0] = Double.parseDouble(row[3]);  // CreditScore
                features[1] = geographyMap.getOrDefault(row[4], -1); // Geography
                features[2] = genderMap.getOrDefault(row[5], -1); // Gender
                features[3] = Double.parseDouble(row[6]);  // Age
                features[4] = Double.parseDouble(row[7]);  // Tenure
                features[5] = Double.parseDouble(row[8]);  // Balance
                features[6] = Double.parseDouble(row[9]);  // NumOfProducts
                features[7] = Double.parseDouble(row[10]); // HasCrCard
                features[8] = Double.parseDouble(row[11]); // IsActiveMember
                features[9] = Double.parseDouble(row[12]); // EstimatedSalary

                featuresList.add(features);
                int target = Integer.parseInt(row[13]); // Exited
                targetList.add(target);

            } catch (NumberFormatException e) {
                System.out.println("Number formatting error in row: " + String.join(",", row));
                e.printStackTrace();
            } catch (Exception e) {
                System.out.println("Error processing row: " + String.join(",", row));
                e.printStackTrace();
            }
        }

        // Convert lists to arrays
        double[][] features = featuresList.toArray(new double[0][]);
        int[] target = targetList.stream().mapToInt(i -> i).toArray(); // Convert target to array

        System.out.println("Features: " + features.length + " rows");
        System.out.println("Target: " + target.length + " entries");
        return features;
    }

    private static int[] extractTarget(List<String[]> data) {
        List<Integer> targetList = new ArrayList<>();
        for (int i = 1; i < data.size(); i++) { // Start from 1 to skip header
            String[] row = data.get(i);
            try {
                int target = Integer.parseInt(row[13]); // Exited
                targetList.add(target);
            } catch (NumberFormatException e) {
                System.out.println("Number formatting error in row: " + String.join(",", row));
                e.printStackTrace();
            }
        }
        return targetList.stream().mapToInt(i -> i).toArray(); // Convert target to array
    }

    private static void splitData(double[][] features, int[] target, double trainFraction) {
        int trainSize = (int) (features.length * trainFraction);
        int testSize = features.length - trainSize;

        double[][] X_train = new double[trainSize][];
        int[] y_train = new int[trainSize];
        double[][] X_test = new double[testSize][];
        int[] y_test = new int[testSize];

        // Create an array of indices
        Integer[] indices = new Integer[features.length];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }

        // Shuffle indices
        Collections.shuffle(Arrays.asList(indices), new Random());

        // Split the data
        for (int i = 0; i < trainSize; i++) {
            X_train[i] = features[indices[i]];
            y_train[i] = target[indices[i]];
        }
        for (int i = 0; i < testSize; i++) {
            X_test[i] = features[indices[trainSize + i]];
            y_test[i] = target[indices[trainSize + i]];
        }

        // Optionally print the sizes of the sets
        System.out.println("Training set size: " + X_train.length);
        System.out.println("Test set size: " + X_test.length);
    }

    private static double[][] normalizeFeatures(double[][] features) {
        int numFeatures = features[0].length;
        double[][] normalizedFeatures = new double[features.length][numFeatures];

        for (int j = 0; j < numFeatures; j++) {
            double sum = 0.0;
            for (double[] feature : features) {
                sum += feature[j];
            }
            double mean = sum / features.length;

            double varianceSum = 0.0;
            for (double[] feature : features) {
                varianceSum += Math.pow(feature[j] - mean, 2);
            }
            double stdDev = Math.sqrt(varianceSum / features.length);

            for (int i = 0; i < features.length; i++) {
                normalizedFeatures[i][j] = (features[i][j] - mean) / stdDev;
            }
        }
        return normalizedFeatures;

        /* Purpose: This function standardizes the features so that they are on a similar scale, which can improve the performance of machine learning algorithms.

        How it works:

        Input: A 2D array of features.
                Steps:

        Initialize Variables:

        numFeatures: The number of features (columns) in the dataset.
        normalizedFeatures: A new 2D array to hold the normalized values.
        Calculate Mean and Standard Deviation:

        For each feature (column), the function calculates the mean and standard deviation.
                The mean is the average value of that feature, and the standard deviation measures how spread out the values are.
        Normalize Each Feature:

        Each feature value is normalized using the formula:

        nv = (value - mean) / std Dev

        This transformation centers the data around zero (mean = 0) and scales it based on the standard deviation.
        Output:

        The function returns the normalized features, which are now on a similar scale, making it easier for the model to learn from the data.


        */

    }

    private static MultiLayerNetwork buildModel(int numInputs, int numOutputs) {
        int numHiddenNodes = 64;

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.001)) // Adam Optimizer
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nIn(numHiddenNodes)
                        .nOut(numOutputs)  //
                        .activation(Activation.SOFTMAX)  // Use softmax for multi-class
                        .lossFunction(LossFunctions.LossFunction.MCXENT)  // cross-entropy loss function
                        .build())
                .build();

        return new MultiLayerNetwork(config);
    }

    /*
    * buildModel(int numInputs, int numOutputs): This function is responsible for constructing the architecture of your neural network. It defines the layers, the number of neurons in each layer, and the activation functions, which determine how the layers interact with the data.

        .seed(123): This sets the random seed for initializing weights in the network. A fixed seed ensures that every
        * time you run the model, the random initialization is the same. This is useful for reproducibility in experiments.

        .updater(new Adam(0.001)): The Adam optimizer is a popular optimization algorithm that adjusts the learning
        * rate dynamically. The 0.001 is the learning rate, controlling how much to adjust the weights of the network
        * with each update. Adam combines the advantages of two other algorithms: AdaGrad (which works well with sparse data)
        * and RMSProp (which works well with non-stationary data).

        DenseLayer.Builder(): This defines a fully connected (dense) layer in the network. Each neuron in this layer is
        * connected to every neuron in the previous and next layers. This is where most of the computation happens.
        * The .nIn(numInputs) specifies how many input neurons this layer expects (the number of features), and .nOut(numHiddenNodes)
        * specifies the number of neurons in this layer.

        nIn(numInputs) / nOut(numHiddenNodes):

        nIn(numInputs): Specifies how many input features the layer should expect. For example, if your data has 10
        * features, you'd set nIn(10).
        nOut(numHiddenNodes): This sets how many neurons (or units) are in this layer. More neurons mean more complexity,
        *  but too many can lead to overfitting.
        Activation.RELU: The Rectified Linear Unit (ReLU) activation function is used in hidden layers to introduce non-linearity.
        *  ReLU outputs the input directly if it’s positive, and zero if it’s negative. It helps the network learn complex patterns by introducing non-linearities.

        OutputLayer.Builder(): This is the final layer of the model. It’s responsible for outputting predictions based
        * on the input data passed through all the previous layers. The activation function here is Activation.SIGMOID,
        * which is often used for binary classification problems, as it outputs values between 0 and 1.

        LossFunctions.LossFunction.XENT: Cross-entropy (XENT) is a loss function used for classification tasks, particularly
        *  for binary classification. It compares the predicted probability distribution (output of the sigmoid) with the actual
        * labels, penalizing incorrect predictions. The goal of training is to minimize this loss.


    *
    *
    * */

}
