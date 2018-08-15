using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MLNET_Practise
{
    public class Iris1
    {
        private string _dataPath;
        private LearningPipeline _pipeline;

        public Iris1(string dataPath)
        {
            this._dataPath = dataPath;

            this.init();
        }

        private void init()
        {
            this._pipeline = new LearningPipeline() {
                new TextLoader(this._dataPath).CreateFrom<IrisData>(separator: ','),
                new Dictionarizer("Label"),
                new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"),
                new StochasticDualCoordinateAscentClassifier(),
                new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" }
            };
        }

        public string Train(IrisData data)
        {
            var model = this._pipeline.Train<IrisData, IrisPrediction>();
            var prediction = model.Predict(data);

            return prediction.PredictedLabels;
        }
    }
}
