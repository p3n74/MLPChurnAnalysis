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

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class MLPChurnAnalysis {

    public static void main(String[] args) {
        List<String[]> data = loadDataSet("C:/Users/firef/Documents/PROGRAMMING/Machine Learning/Data/bank_data.csv");
        double[][] features = prepareData(data);
        double[][] normalizedFeatures = normalizeFeatures(features);

        // Define the number of inputs and outputs
        int numInputs = 10; // Number of features
        int numOutputs = 2; // Binary classification (0 or 1)

        // Build the model
        MultiLayerNetwork model = buildModel(numInputs, numOutputs);

        // Initialize the model
        model.init();

        // Optionally split data into training and testing sets
        double trainFraction = 0.8; // 80% for training
        int[] target = extractTarget(data); // Extract target labels
        splitData(normalizedFeatures, target, trainFraction);

        // Train and evaluate the model
        // For demonstration, we are skipping training and evaluation code
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

        // Skip header row
        boolean firstRow = true;
        for (String[] row : data) {
            if (firstRow) {
                firstRow = false; // Skip the first row (header)
                continue;
            }

            double[] features = new double[10];  // Size matches the number of features we want to keep

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
        for (String[] row : data) {
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
    }

    private static MultiLayerNetwork buildModel(int numInputs, int numOutputs) {
        int numHiddenNodes = 64; // Number of nodes in the hidden layer

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123) // For reproducibility
                .updater(new Adam(0.001)) // Optimizer
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nIn(numHiddenNodes)
                        .nOut(numOutputs)
                        .activation(Activation.SIGMOID)
                        .lossFunction(LossFunctions.LossFunction.XENT) // Cross-Entropy Loss
                        .build())
                .build();

        return new MultiLayerNetwork(config);
    }
}
