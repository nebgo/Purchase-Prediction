using Microsoft.ML.Data;

namespace PurchasePrediction
{
    public class PurchasePrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PurchaseAgain { get; set; }

        [ColumnName("Score")]
        public float Score { get; set; }
    }
}
