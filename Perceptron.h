#pragma once

// Based off the following: https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
class Perceptron
{
public:
	const short predict(const float* aData, const float* aWeights);
	const float* trainWeights(const float* aTrainingData, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const float aLearningRate, const unsigned int aNumberOfEpochs);
private:
};