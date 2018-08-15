using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MLNET_Practise
{
    public class S
    {
        [Column("0")]
        public float S1;
        [Column("1")]
        public float S2;
        [Column("2")]
        public string Label { get; set; }
    }
    public class SPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabels;
    }

    public class SimpleTrain
    {
        private string _dataPath;
        private LearningPipeline _pipeline;

        public SimpleTrain(string dataPath)
        {
            this._dataPath = dataPath;

            this.init();
        }

        private void init()
        {
            this._pipeline = new LearningPipeline() {
                new TextLoader(this._dataPath).CreateFrom<S>(separator: ','),
                new Dictionarizer("Label"),
                new ColumnConcatenator("Features", "S1", "S2"),
                new StochasticDualCoordinateAscentClassifier(),
                new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" }
            };
        }

        public string Train(S data)
        {
            var model = this._pipeline.Train<S, SPrediction>();
            var prediction = model.Predict(data);

            return prediction.PredictedLabels;
        }
    }
}
