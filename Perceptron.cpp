#include <iostream>
#include <algorithm>
#include <time.h>
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
const float* Perceptron::TrainWeights(const float* aTrainingData, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const float aLearningRate, const unsigned int aNumberOfEpochs, bool isPrinting)
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
        if (isPrinting)
        {
            std::cout << "epoch = " << i << ", lrate = " << aLearningRate << ", error = " << sumError << "\n";
        }
    }

    return weights;

}

const float* Perceptron::CrossValidationSplit(const float* aDataset, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const unsigned int aNumberOfFolds)
{
    myFoldSize = aNumberOfDataSamples / aNumberOfFolds;
    int totalNumberOfFoldEntries = (aNumberOfFeatures + 1) * myFoldSize * aNumberOfFolds;
    float* datasetSplit = (float*)malloc(totalNumberOfFoldEntries * sizeof(float));

    int datasetSplitIndex = 0;
    srand(time(0));
    std::vector<int> indiciesUsed = std::vector<int>();
    indiciesUsed.push_back(rand() % aNumberOfDataSamples);
    
    while (indiciesUsed.size() < aNumberOfDataSamples - 1)
    {
        int index = rand() % aNumberOfDataSamples;
        if (std::find(indiciesUsed.begin(), indiciesUsed.end(), index) == indiciesUsed.end())
        {
            indiciesUsed.push_back(index);
            for (int i = 0; i < aNumberOfFeatures + 1; i++)
            {
                float data = aDataset[(index * (aNumberOfFeatures + 1)) + i];
                datasetSplit[datasetSplitIndex] = data;
                std::cout << datasetSplit[datasetSplitIndex] << std::endl;
                datasetSplitIndex++;
            }
        }
    }

    return datasetSplit;
}

const float Perceptron::AccuracyMetric(const float* actual, const float* predicted, const unsigned int aSize)
{
    int correct = 0;
    for (int i = 0; i < aSize; i++)
    {
        if (actual[i] == predicted[i])
        {
            correct++;
        }
    }

    return (correct / (float)aSize) * 100.f;
}

const float* Perceptron::EvaluateScores(const float* aDataset, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const unsigned int aNumberOfFolds, const float aLearningRate, const unsigned int aNumberOfEpochs)
{
    const float* folds = CrossValidationSplit(aDataset, aNumberOfFeatures, aNumberOfDataSamples, aNumberOfFolds);
    
    float* scores = (float*)malloc(aNumberOfFolds * sizeof(float));
    float* trainSet = (float*)malloc((aNumberOfFeatures + 1) * myFoldSize * (aNumberOfFolds - 1) * sizeof(float));
    float* testSet = (float*)malloc((aNumberOfFeatures + 1) * myFoldSize * sizeof(float));
    float* actual = (float*)malloc(myFoldSize * sizeof(float));;

    for (int i = 0; i < aNumberOfFolds; i++)
    {
        // copy fold values into a new array, skipping the current one
        for (int j = 0; j < aNumberOfFolds + 1; j++)
        {
            bool skipped = false;
            if (i != j)
            {
                int index = skipped ? j - 1 : j;
                // copy entire fold ((aNumberOfFeatures + 1) * myFoldSize)
                for(int k = 0; k < (aNumberOfFeatures + 1) * myFoldSize; k++)
                {
                    trainSet[index * (aNumberOfFeatures + 1) * myFoldSize + k] = folds[index * (aNumberOfFeatures + 1) * myFoldSize + k];
                }
            }
            else
            {
                skipped = true;
            }
        }
        
        // copy current fold into testSet
        for (int j = 0; j < (aNumberOfFeatures + 1) * myFoldSize; j++)
        {
            testSet[j] = folds[i * (aNumberOfFeatures + 1) * myFoldSize + j];
        }

        const float* predicted = Execute(trainSet, testSet, aNumberOfFolds, aNumberOfFeatures, aNumberOfDataSamples, aLearningRate, aNumberOfEpochs);
        
        // copy fold predicted values into actual
        for (int j = 0; j < myFoldSize; j++)
        {
            actual[j] = folds[i * (aNumberOfFeatures + 1) * myFoldSize + (j * myFoldSize) + aNumberOfFeatures];
        }

        scores[i] = AccuracyMetric(actual, predicted, myFoldSize);
    }

    return scores;
}

const float* Perceptron::Execute(const float* aTrainingData, const float* aTestData, const unsigned int aNumberOfFolds, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const float aLearningRate, const unsigned int aNumberOfEpochs)
{
    float* predictions = (float*)malloc(aNumberOfFolds * sizeof(float));
    const float* weights = Perceptron::TrainWeights(aTrainingData, aNumberOfFeatures, aNumberOfDataSamples, aLearningRate, aNumberOfEpochs);
    float* testDataRow;

    for (int i = 0; i < myFoldSize; i++)
    {
        testDataRow = (float*)malloc((aNumberOfFeatures + 1) * sizeof(float));
        for (int j = 0; j < (aNumberOfFeatures + 1); j++)
        {
            testDataRow[j] = aTestData[i * (aNumberOfFeatures + 1) + j];
        }
        predictions[i] = Perceptron::Predict(testDataRow, weights, aNumberOfFeatures);
    }

    return predictions;
}