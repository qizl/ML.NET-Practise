using System;

namespace MLNET_Practise
{
    class Program
    {
        static void Main(string[] args)
        {
            var st = new SimpleTrain("st-data.txt");
            var r = st.Train(new S()
            {
                S1 = -1,
                S2 = 12
            });
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
