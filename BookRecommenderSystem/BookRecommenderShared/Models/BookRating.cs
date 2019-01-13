using BookRecommenderShared.Configuration;
using Microsoft.ML.Data;

namespace BookRecommenderShared.Models
{
    public class BookRating
    {
        /* 
         * A machine learning label is the actual data field to be calculated.
         * The ordinal property on the column attribute defines the column number in the dataset.
         */
        [Column(ordinal: "0", name: Labels.Label)]
        public float Label;
        [Column(ordinal: "1")]
        public float user;
        [Column(ordinal: "2")]
        public float bookid;
    }
}
