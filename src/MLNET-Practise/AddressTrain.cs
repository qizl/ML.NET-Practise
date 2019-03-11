using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MLNET_Practise
{

    public class AddressTrain
    {
        private string _dataPath;
        private LearningPipeline _pipeline;

        public AddressTrain(string dataPath)
        {
            this._dataPath = dataPath;

            this.init();
        }

        private void init()
        {
            this._pipeline = new LearningPipeline() {
                new TextLoader(this._dataPath).CreateFrom<address>(separator: ','),
                new Dictionarizer("Label"),
                new ColumnConcatenator("Features", "S1"),
                new StochasticDualCoordinateAscentClassifier(),
                new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" }
            };
        }

        public string Train(float s1)
        {
            var model = this._pipeline.Train<address, sPrediction>();
            var prediction = model.Predict(new address() { S1 = s1 });

            return prediction.PredictedLabels;
        }

        private class address
        {
            [Column("0")]
            public float S1;
            [Column("1")]
            public string Label { get; set; }
        }
        private class sPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }
    }
}
