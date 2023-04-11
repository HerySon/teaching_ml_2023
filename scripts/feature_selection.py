from data_loader import get_data
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
import unittest

class FeatureSelection:
    def __init__(self, df):
        self.df = df

    def set_df(self, df):
        self.df = df

    def ordinal_encoding(self, df): 
        """Encode categorical feats using OrdinalEncoder
        Args:
            df (DataFrame): Dataframe object
        Returns:
            dataframe (DataFrame): output dataframe
        @Author: Nicolas THAIZE
        """
        encoded_df = df.copy()
        ord_enc = OrdinalEncoder()
        categorical_features = encoded_df.select_dtypes('object').columns
        encoded_df[categorical_features] = ord_enc.fit_transform(encoded_df[categorical_features])
        return encoded_df

    def drop_low_var_feats(self, auto_ordinal_encode = True, threshold = 0.1):
        """Drop features that have > (1 - threshold %) simmilar values
        Args:
            df (DataFrame): Dataframe object
            auto_ordinal_encode (boolean): If true, encode categorical features
                By default True
            threshold (float): variance threshold to drop features
        Returns:
            dataframe (DataFrame): output dataframe
        
        @Author: Nicolas THAIZE
        """
        if auto_ordinal_encode:
            self.set_df(self.ordinal_encoding(self.df))
        var_thr = VarianceThreshold(threshold = threshold)
        var_thr.fit(self.df)
        var_thr.get_support()
        low_var_feats = [column for column in self.df.columns if column not in self.df.columns[var_thr.get_support()]]
        return self.df.drop(low_var_feats,axis=1, inplace=True)

if __name__ == "__main__":

    # Unit testing FeatureSelection 
    class TestFeatureSelection(unittest.TestCase):
        def test_shape(self):
            df = get_data(file_path = "../data/en.openfoodfacts.org.products.csv", nrows=50)
            feature_selection = FeatureSelection(df)
            feature_selection.drop_low_var_feats()
            self.assertNotEqual(df.shape, feature_selection.shape, "Shapes are not same size")

        def test_cols(self):
            def isSubset(arr1, arr2):
                m=len(arr1)
                n=len(arr2)
                st = set()
                for i in range(0, m):
                    st.add(arr1[i])
                for i in range(0, n):
                    if arr2[i] in st:
                        continue
                    else:
                        return False
                return True

            df = get_data(file_path = "../data/en.openfoodfacts.org.products.csv", nrows=50)
            feature_selection = FeatureSelection(df)
            feature_selection.drop_low_var_feats()
            assert_val = isSubset(df.columns.to_list(), feature_selection.columns.to_list())

            self.assertTrue(assert_val, "Cols are not included")
    
    unittest.main()
