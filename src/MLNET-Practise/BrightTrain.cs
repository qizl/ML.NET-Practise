using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MLNET_Practise
{
    public class BrightTrain
    {
        private string _dataPath;
        private LearningPipeline _pipeline;

        public BrightTrain(string dataPath)
        {
            this._dataPath = dataPath;

            this.init();
        }

        private void init()
        {
            this._pipeline = new LearningPipeline() {
                new TextLoader(this._dataPath).CreateFrom<b>(separator: ','),
                new Dictionarizer("Label"),
                new ColumnConcatenator("Features", "Time"),
                new StochasticDualCoordinateAscentClassifier(),
                new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" }
            };
        }

        public string Train(float time)
        {
            var model = this._pipeline.Train<b, bPrediction>();
            var prediction = model.Predict(new b() { Time = time });

            return prediction.PredictedLabels;
        }

        private class b
        {
            [Column("0")]
            public float Time;
            [Column("1")]
            public string Label { get; set; }
            //public string Alpha;
        }
        private class bPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }
    }
}
