#include <iostream>
#include "Perceptron.h"

// TODO: update the raw pointers use
constexpr int DATA_SAMPLE_COUNT = 10;
constexpr int NUMBER_OF_FEATURES = 2;

constexpr float DATASET[DATA_SAMPLE_COUNT][NUMBER_OF_FEATURES + 1] = {
    {2.7810836f, 2.550537003f, 0.f},
    {1.465489372f, 2.362125076f, 0.f},
    {3.396561688f, 4.400293529f, 0.f},
    {1.38807019f, 1.850220317f, 0.f},
    {3.06407232f, 3.005305973f, 0.f},
    {7.627531214f, 2.759262235f, 1.f},
    {5.332441248f, 2.088626775f, 1.f},
    {6.922596716f, 1.77106367f, 1.f},
    {8.675418651f, -0.242068655f, 1.f},
    {7.673756466f, 3.508563011f, 1.f}
};

void PredictionDemo()
{
    constexpr float weights[NUMBER_OF_FEATURES + 1] = { -0.1f, 0.20653640140000007f, -0.23418117710000003f };

    for (int i = 0; i < DATA_SAMPLE_COUNT; i++)
    {
        short prediction = Perceptron::Predict(DATASET[i], weights, NUMBER_OF_FEATURES);
        std::cout << "Expected=" << DATASET[i][NUMBER_OF_FEATURES] << "Predicted=" << prediction << "\n";
    }
}

void TrainingNetworkWeightsDemo()
{
    constexpr float LEARNING_RATE = 0.1f;
    constexpr unsigned int NUMBER_OF_EPOCH = 5;
    const float* weights = Perceptron::TrainWeights((float*)DATASET, NUMBER_OF_FEATURES, DATA_SAMPLE_COUNT, LEARNING_RATE, NUMBER_OF_EPOCH);

    std::cout << "[";
    for (int i = 0; i < NUMBER_OF_FEATURES + 1; i++)
    {
        std::cout << weights[i];
        if (i == NUMBER_OF_FEATURES)
        {
            std::cout << "]\n";
        }
        else
        {
            std::cout << ",";
        }
    }
}

void ModelPerceptronDemo()
{

}

int main()
{
    PredictionDemo();
    TrainingNetworkWeightsDemo();
    ModelPerceptronDemo();
}