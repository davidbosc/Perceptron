#include <algorithm>
#include <iostream>
#include "Perceptron.h"

using namespace std;

constexpr int DATA_SAMPLE_COUNT = 10;
constexpr int XOR_DATA_SAMPLE_COUNT = 4;
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

constexpr float XOR_DATASET[DATA_SAMPLE_COUNT][NUMBER_OF_FEATURES + 1] = {
    {0.f, 0.f, 0.f},
    {0.f, 1.f, 1.f},
    {1.f, 0.f, 1.f},
    {1.f, 1.f, 0.f}
};

void PredictionDemo(bool useXor)
{
    if (useXor)
    {
        //XOR based on the following: https://www.geeksforgeeks.org/implementation-of-perceptron-algorithm-for-xor-logic-gate-with-2-bit-binary-input/

        float notWeights[NUMBER_OF_FEATURES] = { 0.5f, -1.f };
        float orWeights[NUMBER_OF_FEATURES + 1] = { -0.5f, 1.f, 1.f };
        float andWeights[NUMBER_OF_FEATURES + 1] = { -1.5f, 1.f, 1.f };

        for (int i = 0; i < XOR_DATA_SAMPLE_COUNT; i++)
        {
            const short andPrediction = Perceptron::Predict(XOR_DATASET[i], andWeights, NUMBER_OF_FEATURES);
            const short orPrediction = Perceptron::Predict(XOR_DATASET[i], orWeights, NUMBER_OF_FEATURES);
            const float temp[NUMBER_OF_FEATURES - 1] = {andPrediction};
            const short notPrediction = Perceptron::Predict(temp, notWeights, NUMBER_OF_FEATURES - 1);

            const float finalX[NUMBER_OF_FEATURES] = { orPrediction , notPrediction };
            const short finalPrediction = Perceptron::Predict(finalX, andWeights, NUMBER_OF_FEATURES);

            cout << "Expected=" << XOR_DATASET[i][NUMBER_OF_FEATURES] << "Predicted=" << finalPrediction << endl;
        }
    }
    else
    {
        constexpr float weights[NUMBER_OF_FEATURES + 1] = { -0.1f, 0.20653640140000007f, -0.23418117710000003f };

        for (int i = 0; i < DATA_SAMPLE_COUNT; i++)
        {
            short prediction = Perceptron::Predict(DATASET[i], weights, NUMBER_OF_FEATURES);
            cout << "Expected=" << DATASET[i][NUMBER_OF_FEATURES] << "Predicted=" << prediction << endl;
        }
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

    free((void*)weights);
}

void ModelPerceptronDemo()
{
    constexpr unsigned int N_FOLDS = 3;
    constexpr float LEARNING_RATE = 0.001f;
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

    free((void*)scores);
}

int main()
{
    bool continueExecuting;
    do
    {
        continueExecuting = 0;
        cout << "****************************************************************************************************" << endl;
        cout << "Enter...\n1 to run PredictionDemo\n2 to run TrainingNetworkWeightsDemo\n3 to run ModelPerceptronDemo" << endl;
        cout << "****************************************************************************************************" << endl;
        int input;
        cin >> input;

        switch (input)
        {
        case 1:
            cout << "\n*************************************************" << endl;
            cout << "Enter...\n0 for sample dataset\n1 for XOR dataset" << endl;
            cout << "*************************************************" << endl;
            bool useXor;
            cin >> useXor;

            PredictionDemo(useXor);
            break;
        case 2:
            TrainingNetworkWeightsDemo();
            break;
        default:
            ModelPerceptronDemo();
            break;
        }

        cout << "\n***********************************" << endl;
        cout << "Enter...\n0 to quit\n1 to run again" << endl;
        cout << "***********************************\n" << endl;
        cin >> continueExecuting;

    } while (continueExecuting);
    
    return 0;
}