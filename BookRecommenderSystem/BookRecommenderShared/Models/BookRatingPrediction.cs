using Microsoft.ML.Data;

namespace BookRecommenderShared.Models
{
    public class BookRatingPrediction
    {
        /* 
         * A machine learning label is the actual data field to be calculated.
         */
        [ColumnName("Label")]
        public float Label;
        /*
         * Score defines the predicted value by the model
         */
        [ColumnName("Score")]
        public float Score;
    }
}
