#pragma once
#include <vector>

// Based off the following: https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/

class Perceptron
{
public:
	static const short Predict(const float* aData, const float* aWeights, const unsigned int aNumberOfFeatures);
	static const float* TrainWeights(const float* aTrainingData, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const float aLearningRate, const unsigned int aNumberOfEpochs, bool isPrinting = false);

	//TODO: replace vectors with pointers - should be 3x3... n_folds x fold_size x number of features + 1
	const float* EvaluateScores(const float* aDataset, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const unsigned int aNumberOfFolds, const float aLearningRate, const unsigned int aNumberOfEpochs);
private:
	const float* CrossValidationSplit(const float* aDataset, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const unsigned int aNumberOfFolds);
	const float AccuracyMetric(const float* actual, const float* predicted, const unsigned int aSize);
	const float* Execute(const float* aTrainingData, const float* aTestData, const unsigned int aNumberOfFolds, const unsigned int aNumberOfFeatures, const unsigned int aNumberOfDataSamples, const float aLearningRate, const unsigned int aNumberOfEpochs);

	int myFoldSize;
};