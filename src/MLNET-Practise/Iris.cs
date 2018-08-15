using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MLNET_Practise
{
    /*
     * 参考：https://www.microsoft.com/net/learn/machine-learning-and-ai/get-started-with-ml-dotnet-tutorial
     */
    // STEP 1: Define your data structures

    // IrisData is used to provide training data, and as 
    // input for prediction operations
    // - First 4 properties are inputs/features used to predict the label
    // - Label is what you are predicting, and is only set when training
    public class IrisData
    {
        [Column("0")]
        public float SepalLength;

        [Column("1")]
        public float SepalWidth;

        [Column("2")]
        public float PetalLength;

        [Column("3")]
        public float PetalWidth;

        [Column("4")]
        [ColumnName("Label")]
        public string Label;
    }

    // IrisPrediction is the result returned from prediction operations
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }

    public class Iris
    {
        private string _dataPath;
        private LearningPipeline _pipeline;

        public Iris(string dataPath)
        {
            this._dataPath = dataPath;

            this.init();
        }

        private void init()
        {
            // STEP 2: Create a pipeline and load your data
            this._pipeline = new LearningPipeline();

            // If working in Visual Studio, make sure the 'Copy to Output Directory' 
            // property of iris-data.txt is set to 'Copy always'
            this._pipeline.Add(new TextLoader(this._dataPath).CreateFrom<IrisData>(separator: ','));

            // STEP 3: Transform your data
            // Assign numeric values to text in the "Label" column, because only
            // numbers can be processed during model training
            this._pipeline.Add(new Dictionarizer("Label"));

            // Puts all features into a vector
            this._pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

            // STEP 4: Add learner
            // Add a learning algorithm to the pipeline. 
            // This is a classification scenario (What type of iris is this?)
            this._pipeline.Add(new StochasticDualCoordinateAscentClassifier());

            // Convert the Label back into original text (after converting to number in step 3)
            this._pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });
        }

        public string Train(IrisData data)
        {
            // STEP 5: Train your model based on the data set
            var model = this._pipeline.Train<IrisData, IrisPrediction>();

            // STEP 6: Use your model to make a prediction
            // You can change these numbers to test different predictions
            var prediction = model.Predict(data);

            return prediction.PredictedLabels;
        }
    }
}
