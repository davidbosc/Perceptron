#include <iostream>
#include "Perceptron.h"

using namespace std;

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
        cout << "Expected=" << DATASET[i][NUMBER_OF_FEATURES] << "Predicted=" << prediction << endl;
    }
}

void TrainingNetworkWeightsDemo()
{
    constexpr float LEARNING_RATE = 0.1f;
    constexpr unsigned int NUMBER_OF_EPOCH = 5;
    const float* weights = Perceptron::TrainWeights((float*)DATASET, NUMBER_OF_FEATURES, DATA_SAMPLE_COUNT, LEARNING_RATE, NUMBER_OF_EPOCH, true);

    cout << "[";
    for (int i = 0; i < NUMBER_OF_FEATURES + 1; i++)
    {
        cout << weights[i];
        if (i == NUMBER_OF_FEATURES)
        {
            cout << "]" << endl;
        }
        else
        {
            cout << ",";
        }
    }
}

void ModelPerceptronDemo()
{
    constexpr unsigned int N_FOLDS = 3;
    constexpr float LEARNING_RATE = 0.1f;
    constexpr unsigned int NUMBER_OF_EPOCH = 500;

    Perceptron p = Perceptron();

    const float* scores = p.EvaluateScores((float*)DATASET, NUMBER_OF_FEATURES, DATA_SAMPLE_COUNT, N_FOLDS, LEARNING_RATE, NUMBER_OF_EPOCH);

    cout << "Scores: [";
    float mean = 0.f;
    for (int i = 0; i < N_FOLDS; i++)
    {
        mean += scores[i];
        cout << scores[i];
        if (i == N_FOLDS - 1)
        {
            cout << "]" << endl;
        }
        else
        {
            cout << ",";
        }
    }
    cout << "Mean Accuracy: " << mean / (float)N_FOLDS << endl;
}

int main()
{
    /*cout << "Enter 1 to run PredictionDemo, 2 to run TrainingNetworkWeightsDemo or anything else to run ModelPerceptronDemo" << endl;
    int input;
    cin >> input;
    
    switch (input)
    {
        case 1:
            PredictionDemo();
            break;
        case 2:
            TrainingNetworkWeightsDemo();
            break;
        default:
            ModelPerceptronDemo();
            break;
    }*/
    ModelPerceptronDemo();
    exit(0);
}