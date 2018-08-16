﻿using System;

namespace MLNET_Practise
{
    class Program
    {
        static void Main(string[] args)
        {
            var ct = new BrightTrain("bt-data.txt");
            var r = ct.Train(1326);
            Console.WriteLine($"result is: {r}");
        }

        static void testCompare()
        {
            var ct = new CompareTrain("ct-data.txt");
            var r = ct.Train(-1, 12);
            Console.WriteLine($"result is: {r}");
        }

        static void testIris()
        {
            var dataPath = "iris-data.txt";

            //var iris = new Iris(dataPath);
            var iris = new Iris1(dataPath);

            var r = iris.Train(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            });

            Console.WriteLine($"Predicted flower type is: {r}");
        }
    }
}
