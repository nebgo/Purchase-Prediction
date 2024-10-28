using Microsoft.ML.Data;

namespace PurchasePrediction
{
    public class PurchaseData
    {
        [LoadColumn(0)]
        public string CustomerID { get; set; }

        [LoadColumn(1)]
        public float AgeCount { get; set; }

        [LoadColumn(2)]
        public string GenderType { get; set; }

        [LoadColumn(3)]
        public float PrepurchaseAmount { get; set; }

        [LoadColumn(4)]
        public bool PurchaseAgain { get; set; } 
    }
}
