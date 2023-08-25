#include <iostream>
#include "Perceptron.h"

int main()
{
    Perceptron p = Perceptron();

    constexpr int DATA_SAMPLE_COUNT = 10;
    constexpr int NUMBER_OF_FEATURES = 2;

    float dataset[DATA_SAMPLE_COUNT][NUMBER_OF_FEATURES + 1] = { 
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

    //float weights[NUMBER_OF_FEATURES + 1] = { -0.1f, 0.20653640140000007f, -0.23418117710000003f };

    /*for (int i = 0; i < DATA_SAMPLE_COUNT; i++)
    {
        short prediction = p.predict(dataset[i], weights);
        std::cout << "Expected=" << dataset[i][NUMBER_OF_FEATURES] << "Predicted=" << prediction << "\n";
    }*/

    const float learningRate = 0.1f;
    const unsigned int numberOfEpoch = 5;
    const float* weights = p.trainWeights((float*)dataset, NUMBER_OF_FEATURES, DATA_SAMPLE_COUNT, learningRate, numberOfEpoch);

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