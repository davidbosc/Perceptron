#pragma once
#include <vector>

// Based off the following: https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/

class Perceptron
{
public:
	static const short Predict(const float* aData, const float* aWeights, const unsigned int aNumberOfFeatures);
	static const float* TrainWeights(const float* aTrainingData, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const float aLearningRate, const unsigned int aNumberOfEpochs);

	//TODO: replace vectors with pointers - should be 3x3... n_folds x fold_size x number of features + 1
	std::vector<float> EvaluateScores(const float* aDataset, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const unsigned int aNumberOfFolds,
		std::vector<float>(*func)(const float* aTrainingData, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const float aLearningRate, const unsigned int aNumberOfEpochs));
private:
	std::vector<std::vector<float>> CrossValidationSplit(const float* aDataset, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const unsigned int aNumberOfFolds);
	const float* CrossValidationSplit(const float* aDataset, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const unsigned int aNumberOfFolds);
	const float AccuracyMetric(std::vector<float> actual, std::vector<float> predicted);
};