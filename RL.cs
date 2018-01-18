using UnityEngine;
using System.Collections.Generic;
using UnityEngine.UI;
using System.IO;

public class RL : MonoBehaviour
{
    public struct Transition
    {
        public float[] state;
        public int action;
        public float reward;
    };

    NN neuralNetwork;

    public Transform[] targets;
    public Image[] images;
    
    public Text epsilonText;
    float epsilon = 1;

    int trainSteps = 0;

    private void Start()
    {
        neuralNetwork = new NN(14, 10, 66);
    }

    public int Act(float[] state)
    {
        List<float> output = neuralNetwork.calcNet(state);
        int maxIndex = 0;
        float max = output[0], min = output[0];
        for(int i = 1; i < output.Count; i++)
        {
            if(output[i] > max)
            {
                maxIndex = i;
                max = output[i];
            }
            else if (output[i] < min)
            {
                min = output[i];
            }
        }
        for (int i = 0; i < output.Count; i++)
        {
            float ratio = (output[i] - min) / (max - min);
            images[i].color = new Color(1 - ratio, ratio, 0, 0.25f);
        }
        images[maxIndex].color = new Color(0, 1, 0, 0.5f);
        return Random.Range(0.0f, 1.0f) >= epsilon ? maxIndex : Random.Range(0, targets.Length);
    }

    public void Observe(Transition newTransition)
    {
        neuralNetwork.Train(newTransition);
        trainSteps++;
        if(trainSteps % 100 == 0 && epsilon > 0.001f)
        {
            epsilon -= 0.01f;
            epsilonText.text = "Epsilon = " + epsilon.ToString("0.00");
        }
    }

    public void Save(StreamWriter writer)
    {
        writer.WriteLine(epsilon);
        writer.WriteLine(trainSteps);
        neuralNetwork.Save(writer);
    }

    public void Load(StreamReader reader)
    {
        try
        {
            epsilon = float.Parse(reader.ReadLine());
            trainSteps = int.Parse(reader.ReadLine());
            neuralNetwork.Load(reader);
        }
        catch (System.Exception e)
        {
            Debug.Log(e.Message);
        }
        if (epsilon < 0.01f)
            epsilon = 0;
        epsilonText.text = "Epsilon = " + epsilon.ToString("0.00");
    }
}