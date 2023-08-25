#include <iostream>
#include "Perceptron.h"

const short Perceptron::predict(const float* aData, const float* aWeights)
{
    float activation = aWeights[0];
    int rowSize = sizeof(aData) / sizeof(aData[0]);

    for (int i = 0; i < rowSize; i++)
    {
        activation += aWeights[i + 1] * aData[i];
    }

    return activation >= 0.0f ? 1.f : 0.f;
}

// stochastic gradient descent
const float* Perceptron::trainWeights(const float* aTrainingData, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const float aLearningRate, const unsigned int aNumberOfEpochs)
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

            float prediction = predict(data, weights);
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