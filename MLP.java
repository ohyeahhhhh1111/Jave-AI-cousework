import java.io.BufferedReader;
import java.io.FileReader; //To read csv file.　
import java.io.IOException; //It is used for debugging.
import java.util.Arrays; //Array package.


public class MLPExample {
	
    public static double[][][] loadCSV(String filePath, int numExamples, int numFeatures) throws IOException {
        //運用 bufferreader 套件建立reader物件
    	// Use the BufferedReader package to create a reader object.
    	BufferedReader reader = new BufferedReader(new FileReader(filePath));
        //將inputs以及targets陣列初始化
    	// Initialize the inputs and targets arrays.
    	double[][] inputs = new double[numExamples][numFeatures];
        double[][] targets = new double[numExamples][10]; 
        //inputs為輸入的檔案，targets為數字0~9的機率
        // inputs represent the input file, and targets represent the probabilities for numbers 0 to 9.
        
        String line;
        int dataIndex = 0;
        //新增dataIndex用來記錄用了幾個數據的數量
        //Add dataIndex to record the number of data entries used.
        while ((line = reader.readLine()) != null && dataIndex < numExamples) {
            //將每個特徵split出來
        	// Split the features individually.
        	String[] parts = line.split(",");
            
        	//將每個特徵放進去inputs裡面
        	// Put each feature into the inputs array.
            for (int FeaturesIndex = 0; FeaturesIndex < numFeatures; FeaturesIndex++) {
                inputs[dataIndex][FeaturesIndex] = Double.parseDouble(parts[FeaturesIndex]) / 16;
            }
            
            //將label也放進去然後做成targets，假設數字為2則給出target為機率所以就是0,0,1,0,0...
            //Add labels into targets. For example, if the number is 2, the target is represented as probabilities: 0, 0, 1, 0, 0...
            int label = Integer.parseInt(parts[numFeatures]);
            for (int labIndex = 0; labIndex < 10; labIndex++) {
                targets[dataIndex][labIndex] = (labIndex == label) ? 1.0 : 0.0;
            }
            dataIndex++;
        }

        return new double[][][]{inputs, targets};
    }

    // Store the best weights and biases.
    private static double[][] best_ipt2hiddenWeights; 
    private static double[][] best_hidden2optWeights;
    private static double[] best_ipt2hiddenbias; 
    private static double[] best_hidden2optbias;
    private static double best_Accuracy = -999;

    public static class ImprovedMLP {
    	//建立輸入大小、隱藏層大小、輸出大小
    	//建立LearningRate用來更新weight以及bias
    	// Create inputSize, hiddenSize, and outputSize.
    	// Create LearningRate to update weights and biases.
        private int inputSize;
        private int hiddenSize;
        private int outputSize;
        private double learningRate;//to update weight and bias
        private int step;
        
        //建立出所有weight以及所有bias
        // Create all weights and all biases.
        private double[][] ipt2hiddenWeights; 
        private double[][] hidden2optWeights; 
        private double[] ipt2hiddenbias; 
        private double[] hidden2optbias;

        //建立感知機物件
        // Create an MLP object.
        public ImprovedMLP(int inputSize, int hiddenSize, int outputSize, double learningRate, int step) {
            //物件的建構子
        	// Constructor of the MLP object.
        	this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;
            this.learningRate = learningRate;
            this.step = step;

            //將weight以及bias放進去
            // Put the weights and biases into the model.
            ipt2hiddenWeights = new double[inputSize][hiddenSize];
            hidden2optWeights = new double[hiddenSize][outputSize];
            ipt2hiddenbias = new double[hiddenSize];
            hidden2optbias = new double[outputSize];
        }
        
        //初始化weight的程式碼，設為亂數，每次訓練都有可能找到更好的解比起初始化為0
        // Initialize the weights with random values. This allows the model to potentially find better solutions compared to initializing them as zeros during each training.
        private void initializeWeights() {
            for (int iptWeight = 0; iptWeight < inputSize; iptWeight++) {
                for (int ipt2hidden = 0; ipt2hidden < hiddenSize; ipt2hidden++) {
                    ipt2hiddenWeights[iptWeight][ipt2hidden] = Math.random() * Math.sqrt(2.0 / (inputSize + hiddenSize));
                }
            }
            for (int hiddenWeight = 0; hiddenWeight < hiddenSize; hiddenWeight++) {
                for (int hidden2opt = 0; hidden2opt < outputSize; hidden2opt++) {
                    hidden2optWeights[hiddenWeight][hidden2opt] = Math.random() * Math.sqrt(2.0 / (hiddenSize + outputSize));
                }
            }
        }

        
        //第一個forward，從input到hidden的計算
        // The first forward pass, calculating from input to hidden layer.
        private double[] forward(double[] input, double[] output, boolean HidOrOut) {
            if (HidOrOut) {
                for (int optindex = 0; optindex < hiddenSize; optindex++) {
                    output[optindex] = 0;
                    for (int iptIndex = 0; iptIndex < inputSize; iptIndex++) {
                        output[optindex] += input[iptIndex] * ipt2hiddenWeights[iptIndex][optindex];
                    }
                    output[optindex] += ipt2hiddenbias[optindex];
                    output[optindex] = 1 / (1 + Math.exp(-output[optindex]));
                }
            }
            else {
                for (int optindex = 0; optindex < outputSize; optindex++) {
                    output[optindex] = 0;
                    for (int iptIndex = 0; iptIndex < hiddenSize; iptIndex++) {
                        output[optindex] += input[iptIndex] * hidden2optWeights[iptIndex][optindex];
                    }
                    output[optindex] += hidden2optbias[optindex];
                    output[optindex] = 1 / (1 + Math.exp(-output[optindex]));
                }
            }

            return output;
        }


        //做交叉商損失函數來計算出error以利後續更新前面的weight跟bias
        // Use CrossEntropyLoss to calculate the error for updating weights and biases.
        public double[] crossEntropyLoss(double[] output, double[] target) {
            double[] error = new double[outputSize + 1];
        
            for (int errorIndex = 0; errorIndex < error.length - 1; errorIndex++) {
                error[error.length - 1] -= target[errorIndex] * Math.log(Math.max(output[errorIndex], 1e-15)); 
                error[errorIndex] = output[errorIndex] - target[errorIndex];
            }
            return error;
        }
        // 使用 L2 Loss (歐幾里得損失)
        //use L2 Loss to calculate Loss for updating weights and bias
        public double[] l2Loss(double[] output, double[] target) {
            double[] error = new double[outputSize + 1];
            double loss = 0.0;

            for (int errorIndex = 0; errorIndex < outputSize; errorIndex++) {
                double diff = output[errorIndex] - target[errorIndex];
                loss += diff * diff; 
                // 累積平方差 Cumulative squared error.
                error[errorIndex] = diff; 
                //將每個差值記錄為誤差
                // Record each difference as an error.
            }
            error[outputSize] = loss / 2.0; // 損失值為平方差的平均（除以 2 保持與梯度一致性）
            return error;
        }

        //反向傳播
        //backpropagation
        private double backward(double[] input, double[] hidden, double[] output, double[] target) {
            
            double[] outputError = crossEntropyLoss(output, target); 
            double loss = outputError[outputError.length - 1];
            //計算出第二個weight以及bias的更新值
            // Calculate the update values for the second hidden layer's weights and biases.
            outputError = Arrays.copyOfRange(outputError, 0, outputError.length - 1);
            
            //計算出第一個weight以及bias的更新值
            // Calculate the update values for the first hidden layer's weights and biases.
            double[] hiddenError = new double[hiddenSize];
            for (int hidErrorIndex = 0; hidErrorIndex < hidden.length; hidErrorIndex++) {
                hiddenError[hidErrorIndex] = 0;
                for (int optErrorIndex = 0; optErrorIndex < outputError.length; optErrorIndex++) {
                    hiddenError[hidErrorIndex] += outputError[optErrorIndex] * hidden2optWeights[hidErrorIndex][optErrorIndex];
                }
                hiddenError[hidErrorIndex] *= hidden[hidErrorIndex] * (1 - hidden[hidErrorIndex]);
            }

            //對第二個weight以及bias更新
            // Update the second hidden layer's weights and biases.
            for (int hidIndex = 0; hidIndex < hidden.length; hidIndex++) {
                for (int optweightsIndex = 0; optweightsIndex < outputError.length; optweightsIndex++) {
                    hidden2optWeights[hidIndex][optweightsIndex] -= learningRate * hidden[hidIndex] * outputError[optweightsIndex];
                }
            }
            for (int optbias = 0; optbias < outputError.length; optbias++) {
                hidden2optbias[optbias] -= learningRate * outputError[optbias];
            }
            
            //對第一個weights以及bias做更新
            // Update the first hidden layer's weights and biases.
            for (int iptIndex = 0; iptIndex < input.length; iptIndex++) {
                for (int hidweightsIndex = 0; hidweightsIndex < hiddenError.length; hidweightsIndex++) {
                    ipt2hiddenWeights[iptIndex][hidweightsIndex] -= learningRate * input[iptIndex] * hiddenError[hidweightsIndex];
                }
            }
            for (int hidbias = 0; hidbias < hiddenError.length; hidbias++) {
                ipt2hiddenbias[hidbias] -= learningRate * hiddenError[hidbias];
            }
            
            //最後輸出loss，loss是一種模型自信度的表示值
            // Output the loss at the end.
            // Loss represents the model's confidence level.
            return loss;
        }

        //一個函數，讓我將模型產出的0~9的機率拿去對答案
        // Create a function to generate probabilities for digits 0 to 9 and compare with the correct answer.
        private int predict(double[] output) {
            int maxIdx = 0;
            for (int optindex = 1; optindex < outputSize; optindex++) {
                if (output[optindex] > output[maxIdx]) {
                    maxIdx = optindex;
                }
            }
            return maxIdx;
        }

        //訓練的程式碼，將前面的所有模型相關的函數整合，並且將正確率計算出來
        //epoch為跌代次數，拿來訓練更多次，好讓模型的產出更擬合答案
        //the code of training.It fuse all above, and calculate the accuracy
        //epoch is for train more then once.It can make model fit the target.
        public void train(double[][] trainInputs, double[][] trainTargets, double[][] valInputs, double[][] valTargets, int epochs, int batchSize) {
            double[] hidden = new double[hiddenSize];
            double[] output = new double[outputSize];
            for (int epoch = 1; epoch <= epochs; epoch++) {
                if (epoch % step == 0) {
                    learningRate *= 0.1;
                }

                double totalLoss = 0;
                int correctPredictions = 0;

                // batch training. I hope it can increase the generalization of model.
                for (int batchStart = 0; batchStart < trainInputs.length; batchStart += batchSize) {
                    int batchEnd = Math.min(batchStart + batchSize, trainInputs.length);

                    for (int batchIndex = batchStart; batchIndex < batchEnd; batchIndex++) {
                        double[] input = trainInputs[batchIndex];
                        double[] target = trainTargets[batchIndex];
                        hidden = forward(input, hidden, true);
                        output = forward(hidden, output, false);
                        if (predict(output) == predict(target)) {
                            correctPredictions++;
                        }
                        totalLoss += backward(input, hidden, output, target);
                    }
                }

                // calculate the accuracy of training data.
                double trainAccuracy = (double) correctPredictions / trainInputs.length * 100;
                int valCorrectPredictions = 0;
                for (int valIndex = 0; valIndex < valInputs.length; valIndex++) {
                    double[] input = valInputs[valIndex];
                    double[] target = valTargets[valIndex];
                    hidden = forward(input, hidden, true);
                    output = forward(hidden, output, false);
                    
                    if (predict(output) == predict(target)) valCorrectPredictions++;
                }
                // calculate the accuracy of Validation data.
                double valAccuracy = (double) valCorrectPredictions / valInputs.length * 100;
                // If we get a better result, recording it.
                if (best_Accuracy < valCorrectPredictions) {
                    best_ipt2hiddenWeights = ipt2hiddenWeights.clone(); 
                    best_hidden2optWeights = hidden2optWeights.clone();
                    best_ipt2hiddenbias = ipt2hiddenbias.clone(); 
                    best_hidden2optbias = hidden2optbias.clone();
                    best_Accuracy = valCorrectPredictions;
                }
                else {
                	
                }

                System.out.printf("Epoch %d - Train Loss: %.4f, Train Accuracy: %.2f%%, Val Accuracy: %.2f%%\n", epoch, totalLoss / trainInputs.length, trainAccuracy, valAccuracy);
            }
        }


        //將資料做test
        // After training, test the entire dataset and calculate the accuracy.
        public void test(double[][] trainInputs, double[][] trainTargets, double[][] valInputs, double[][] valTargets) {
            int testCorrectPredictions = 0;
            int correctPredictions = 0;
            double[] hidden = new double[hiddenSize];
            double[] output = new double[outputSize];
            for (int trainIndex = 0; trainIndex < trainInputs.length; trainIndex++) {
                double[] input = trainInputs[trainIndex];
                double[] target = trainTargets[trainIndex];
                hidden = forward(input, hidden, true);
                output = forward(hidden, output, false);

                if (predict(output) == predict(target)) correctPredictions++;
            }
            for (int valIndex = 0; valIndex < valInputs.length; valIndex++) {
                double[] input = valInputs[valIndex];
                double[] target = valTargets[valIndex];
                hidden = forward(input, hidden, true);
                output = forward(hidden, output, false);
                if (predict(output) == predict(target)) testCorrectPredictions++;
            }
            double trainAccuracy = (double) correctPredictions / trainInputs.length * 100;
            double testAccuracy = (double) testCorrectPredictions / valInputs.length * 100;
            System.out.printf("Best Train Accuracy: %.2f%%\n", trainAccuracy);
            System.out.printf("Best Test Accuracy: %.2f%%\n", testAccuracy);
            System.out.printf("Best Average Accuracy: %.2f%%\n", ((trainAccuracy + testAccuracy) / 2.0));
        }
    }

    public static void main(String[] args) throws IOException {
        //輸入的資料特徵長度，輸入資料為1*65而最後一個為target所以長度為64不可改變
        //input size for 1*64 data
    	int inputSize = 64;
    	//隱藏層的長度，越長可以有更多的weight以及bias
        //the length of Hidden layer.the number getting bigger and the model getting more weights
        int hiddenSize = 128;
        //輸出的長度，不可以改，就是模型輸出0~9每個數字的機率
        //output size.It must be 10,because it represent the probabilities between 0 to 9.
        int outputSize = 10;
        //LearningRate學習率的概念，可以隨意調整，為更新weight以及bias的幅度
        //The learning rate determines the magnitude of updates to the weights and biases.
        double learningRate = 0.1314;
        //stpe可以調整更新Learning的epoch
        //step can update Learning Rate when the epochs equal to it.
        int step =450;
        //epoch跌代次數
        //the times that I train model.
        int epoch = 500;
        //新增batch size來使用批次訓練
        //use it to change the batch when the model is training.
        int batch_size = 32;
        
        //建立出物件
        //build a object ImprovedMLP and it name is mlp.
        ImprovedMLP mlp = new ImprovedMLP(inputSize, hiddenSize, outputSize, learningRate, step);

        //建立train的data
        //create training data
        double[][][] trainData = loadCSV("src/dataSet1.csv", 2810, inputSize);
        double[][] trainInputs = trainData[0];
        double[][] trainTargets = trainData[1];

        //建立test的data
        //create Validation data.
        double[][][] valData = loadCSV("src/dataSet2.csv", 2810, inputSize);
        double[][] valInputs = valData[0];
        double[][] valTargets = valData[1];

        //訓練
        //train.
        mlp.initializeWeights();
        mlp.train(trainInputs, trainTargets, valInputs, valTargets, epoch, batch_size);
        
        // Load the best model and test the results.
        mlp.ipt2hiddenWeights = best_ipt2hiddenWeights.clone();
        mlp.hidden2optWeights = best_hidden2optWeights.clone();
        mlp.ipt2hiddenbias = best_ipt2hiddenbias.clone();
        mlp.hidden2optbias = best_hidden2optbias.clone();
        
        mlp.test(trainInputs, trainTargets, valInputs, valTargets);  
    }
}
