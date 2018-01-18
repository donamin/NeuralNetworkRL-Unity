using UnityEngine;
using System.Collections.Generic;
using System.IO;

public class NN
{
    //user defineable variables
    int numEpochs;
    //number of inputs - this includes the input bias
    int numInputs = 0;
    //number of hidden units
    int numHiddens;
    //number of output units
    int numOutputs;
    //Size of experience replay memory!
    int sizeOfExperienceReplayMemory;
    //learning rate
    float LR;

    int batchSize;

    //training data
    List<RL.Transition> transitions, batchTransitions;

    //the outputs of the neurons
    float[] hiddenVal;
    float[] outputVal;

    //the errors
    float[] hiddenDelta;
    float[] outputDelta;

    //the weights
    float[,] weightsIH;
    float[,] weightsHO;
    
    public NN(int inputs, int hiddens, int outputs)
    {
        transitions = new List<RL.Transition>();
        batchTransitions = new List<RL.Transition>();
        numInputs = inputs;
        numHiddens = hiddens;
        numOutputs = outputs;
        numEpochs = 50;
        sizeOfExperienceReplayMemory = 500;
        LR = 0.01f;
        batchSize = 25;
        InitWeights();
    }

    void InitWeights()
    {
        hiddenVal = new float[numHiddens];
        outputVal = new float[numOutputs];
        hiddenDelta = new float[numHiddens];
        outputDelta = new float[numOutputs];
        weightsIH = new float[numInputs, numHiddens];
        weightsHO = new float[numHiddens + 1, numOutputs];
        for (int j = 0; j < numHiddens; j++)
        {
            for (int i = 0; i < numInputs; i++)
                weightsIH[i, j] = (Random.Range(0.0f, 1.0f) - 0.5f) / 10;
            for (int i = 0; i < numOutputs; i++)
                weightsHO[j, i] = (Random.Range(0.0f, 1.0f) - 0.5f) / 10;
        }
        for (int i = 0; i < numOutputs; i++)
            weightsHO[numHiddens, i] = (Random.Range(0.0f, 1.0f) - 0.5f) / 10;
    }

    float sigmoid(float x)
    {
        return 1.0f / (1 + (float)Mathf.Exp((float)-x));
    }

    public List<float> calcNet(float[] input)
    {
        //calculate the outputs of the hidden neurons
        for (int i = 0; i < numHiddens; i++)
        {
            hiddenVal[i] = 0;
            for (int j = 0; j < numInputs; j++)
                hiddenVal[i] += input[j] * weightsIH[j, i];
            hiddenVal[i] = sigmoid(hiddenVal[i]);
        }
        //calculate the output of the network
        List<float> outPred = new List<float>();
        for (int j = 0; j < numOutputs; j++)
        {
            outputVal[j] = 0;
            for (int i = 0; i < numHiddens; i++)
                outputVal[j] += hiddenVal[i] * weightsHO[i, j];
            outputVal[j] += 1 * weightsHO[numHiddens, j];
            outputVal[j] = sigmoid(outputVal[j]);
            outPred.Add(outputVal[j]);
        }
        return outPred;
    }

    public void Train(RL.Transition newTransition)
    {
        while (transitions.Count >= sizeOfExperienceReplayMemory)
            transitions.RemoveAt(0);
        transitions.Add(newTransition);
        TrainNetwork();
    }

    bool SampleBatch()
    {
        batchTransitions.Clear();
        if (transitions.Count < 2 * batchSize)
            return false;
        for (int i = 0; i < Mathf.Min(batchSize, transitions.Count); i++)
            batchTransitions.Add(transitions[Random.Range(0, transitions.Count)]);
        return true;
    }

    void TrainNetwork()
    {
        if (!SampleBatch())
            return;
        for (int e = 0; e < numEpochs; e++)
        {
            for (int t = 0; t < batchTransitions.Count; t++)
            {
                float target = batchTransitions[t].reward;
                float output = calcNet(batchTransitions[t].state)[batchTransitions[t].action];
                //calculate the error
                outputDelta[batchTransitions[t].action] = (target - output) * output * (1 - output);
                //change network weights
                WeightChangesHO(batchTransitions[t].action, output);
                WeightChangesIH(batchTransitions[t], output);
            }
        }
    }

    //adjust the weights hidden-output
    void WeightChangesHO(int action, float outPred)
    {
        for (int k = 0; k <= numHiddens; k++)
        {
            float gradient = outputDelta[action] * 1; //This is for bias weight
            if (k < numHiddens)
                gradient = outputDelta[action] * hiddenVal[k];
            weightsHO[k, action] += LR * gradient;
        }
    }

    //adjust the weights input-hidden
    void WeightChangesIH(RL.Transition curTransition, float outPred)
    {
        for (int h = 0; h < numHiddens; h++)
        {
            hiddenDelta[h] = weightsHO[h, curTransition.action] * outputDelta[curTransition.action];
            hiddenDelta[h] *= hiddenVal[h] * (1 - hiddenVal[h]); //Derivative for sigmoid function
            for (int i = 0; i < numInputs; i++)
            {
                float gradient = hiddenDelta[h] * curTransition.state[i];
                weightsIH[i, h] += LR * gradient;
            }
        }
    }

    public void Save(StreamWriter writer)
    {
        writer.WriteLine(numEpochs);
        writer.WriteLine(numInputs);
        writer.WriteLine(numHiddens);
        writer.WriteLine(numOutputs);
        writer.WriteLine(sizeOfExperienceReplayMemory);
        writer.WriteLine(LR);
        writer.WriteLine(LR);
        for (int i = 0; i < numInputs; i++)
            for (int h = 0; h < numHiddens; h++)
            {
                writer.WriteLine(weightsIH[i, h]);
            }
        for (int h = 0; h <= numHiddens; h++)
            for (int o = 0; o < numOutputs; o++)
            {
                writer.WriteLine(weightsHO[h, o]);
            }
    }

    public void Load(StreamReader reader)
    {
        try
        {
            numEpochs = int.Parse(reader.ReadLine());
            numInputs = int.Parse(reader.ReadLine());
            numHiddens = int.Parse(reader.ReadLine());
            numOutputs = int.Parse(reader.ReadLine());
            InitWeights();
            sizeOfExperienceReplayMemory = int.Parse(reader.ReadLine());
            LR = float.Parse(reader.ReadLine());
            LR = float.Parse(reader.ReadLine());
            hiddenVal = new float[numHiddens];
            weightsIH = new float[numInputs, numHiddens];
            for (int i = 0; i < numInputs; i++)
                for (int h = 0; h < numHiddens; h++)
                {
                    weightsIH[i, h] = float.Parse(reader.ReadLine());
                }
            weightsHO = new float[numHiddens + 1, numOutputs];
            for (int h = 0; h <= numHiddens; h++)
                for (int o = 0; o < numOutputs; o++)
                {
                    weightsHO[h, o] = float.Parse(reader.ReadLine());
                }
        }
        catch (System.Exception e)
        {
            Debug.Log(e.Message);
        }
    }
}