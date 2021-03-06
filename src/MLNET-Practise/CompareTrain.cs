﻿using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MLNET_Practise
{

    public class CompareTrain
    {
        private string _dataPath;
        private LearningPipeline _pipeline;

        public CompareTrain(string dataPath)
        {
            this._dataPath = dataPath;

            this.init();
        }

        private void init()
        {
            this._pipeline = new LearningPipeline() {
                new TextLoader(this._dataPath).CreateFrom<s>(separator: ','),
                new Dictionarizer("Label"),
                new ColumnConcatenator("Features", "S1", "S2"),
                new StochasticDualCoordinateAscentClassifier(),
                new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" }
            };
        }

        public string Train(float s1, float s2)
        {
            var model = this._pipeline.Train<s, sPrediction>();
            var prediction = model.Predict(new s() { S1 = s1, S2 = s2 });

            return prediction.PredictedLabels;
        }

        private class s
        {
            [Column("0")]
            public float S1;
            [Column("1")]
            public float S2;
            [Column("2")]
            public string Label { get; set; }
        }
        private class sPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }
    }
}
