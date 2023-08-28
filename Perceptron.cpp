#include <iostream>
#include <algorithm>
#include "Perceptron.h"

const short Perceptron::Predict(const float* aData, const float* aWeights, const unsigned int aNumberOfFeatures)
{
    float activation = aWeights[0];

    for (int i = 0; i < aNumberOfFeatures; i++)
    {
        activation += aWeights[i + 1] * aData[i];
    }

    return activation >= 0.0f ? 1.f : 0.f;
}

// stochastic gradient descent
const float* Perceptron::TrainWeights(const float* aTrainingData, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const float aLearningRate, const unsigned int aNumberOfEpochs)
{
    float* weights = (float*) malloc((aNumberOfFeatures + 1) * sizeof(float));

    // memset didn't work?
    for (int i = 0; i < aNumberOfFeatures + 1; i++)
    {
        weights[i] = 0.f;
    }

    for (int i = 0; i < aNumberOfEpochs; i++)
    {
        float sumError = 0.f;
        for (int j = 0; j < aNumberOfDataSamples; j++)
        {
            float* data = (float*) malloc((aNumberOfFeatures + 1) * sizeof(float));

            // memset didn't work?
            for (int k = 0; k < aNumberOfFeatures + 1; k++)
            {
                data[k] = aTrainingData[j * (aNumberOfFeatures + 1) + k];
            }

            float prediction = Predict(data, weights, aNumberOfFeatures);
            float error = aTrainingData[j * (aNumberOfFeatures + 1) + aNumberOfFeatures ] - prediction;
            sumError += std::powf(error, 2.f);
            weights[0] = weights[0] + aLearningRate * error;
            for (int k = 0; k < aNumberOfFeatures; k++)
            {
                weights[k + 1] = weights[k + 1] + aLearningRate * error * aTrainingData[j * (aNumberOfFeatures + 1) + k];
            }
        }
        std::cout << "epoch = " << i << ", lrate = " << aLearningRate << ", error = " << sumError << "\n";
    }

    return weights;

}

std::vector<std::vector<float>> Perceptron::CrossValidationSplit(const float* aDataset, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const unsigned int aNumberOfFolds)
{
    std::vector<std::vector<float>> aDatasetSplit = std::vector<std::vector<float>>();

    unsigned int foldSize = aNumberOfDataSamples / aNumberOfFolds;

    for (int i = 0; i < aNumberOfFolds; i++)
    {
        std::vector<float> fold = std::vector<float>();
        while (fold.size() < foldSize)
        {
            int index = rand() % aNumberOfDataSamples;
            for (int i = 0; i < aNumberOfFeatures + 1; i++)
            {
                float data = aDataset[(index * (aNumberOfFeatures + 1)) + i];
                fold.push_back(data);
            }
        }
        aDatasetSplit.push_back(fold);
    }
    
    return aDatasetSplit;
}

const float Perceptron::AccuracyMetric(std::vector<float> actual, std::vector<float> predicted)
{
    int correct = 0;
    for (int i = 0; i < actual.size(); i++)
    {
        if (actual[i] == predicted[i])
        {
            correct++;
        }
    }

    return (correct / (float)actual.size()) * 100.f;
}

std::vector<float> Perceptron::EvaluateScores(const float* aDataset, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const unsigned int aNumberOfFolds, std::vector<float>(*func)(const float* aTrainingData, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const float aLearningRate, const unsigned int aNumberOfEpochs))
{
//    std::vector<std::vector<float>> folds = CrossValidationSplit(aDataset, aNumberOfFeatures, aNumberOfDataSamples, aNumberOfFolds);
//    std::vector<float> scores = std::vector<float>();
//
//    for (int i = folds.size() - 1; i >= 0; i--)
//    {
//        std::vector<std::vector<float>> trainSets = std::vector<std::vector<float>>(folds);
//        if (i < folds.size() - 1)
//        {
//            std::swap(trainSets[i], trainSets[folds.size() - 1]);
//        }
//        trainSets.pop_back();
//        // put each element into its own list
//
//        //trainSets.size() * trainSets[0].size()?
//        float* test = (float*)malloc((aNumberOfFeatures + 1) * aNumberOfDataSamples * sizeof(float));
//        for (int j = 0; j < trainSets.size(); j++)
//        {
//            for (int k = 0; k < trainSets[j].size(); k++)
//            {
//                test[j * trainSets[j].size() + k] = trainSets[j][k];
//            }
//        }
//
//        std::vector<float> test_set = std::vector<float>();
//        for (float row : trainSets[i])
//        {
////            row_copy = list(row)
//            std::vector<float> copy = std::vector<float>(row);
//            test_set.append(row_copy)
////            row_copy[-1] = None
//        }
//    }
    
    return std::vector<float>();
}

//def evaluate_algorithm(dataset, algorithm, n_folds, *args) :
//        for row in fold :
//            row_copy = list(row)
//            test_set.append(row_copy)
//            row_copy[-1] = None
//        predicted = algorithm(train_set, test_set, *args)
//        actual = [row[-1] for row in fold]
//        accuracy = accuracy_metric(actual, predicted)
//        scores.append(accuracy)
//    return scores