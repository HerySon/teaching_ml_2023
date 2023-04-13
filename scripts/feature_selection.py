from data_loader import get_data
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
import unittest
import numpy as np
import matplotlib.pyplot as plt
from fsfc.generic import NormalizedCut
from sklearn.metrics.pairwise import rbf_kernel

class FeatureSelection:
    def __init__(self, df):
        self.df = df

    def set_df(self, df):
        """Set df value provided in arg
        Args:
            df (DataFrame): Dataframe object
        @Author: Nicolas THAIZE
        """
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
    
    def plot_dispertion_ratio(self, auto_ordinal_encode=True, threshold=1, plot_y_scale="log"):
        """Plot dispertion ratio calculus on whole dataset 
        Args:
            auto_ordinal_encode (boolean, optional): If true, encode categorical features
                By default True
            threshold (integer, optional): min dispertion ratio value to display on plot
                By default 1
            plot_y_scale (string, optional): Change plot y axis scale
                By default log
        @Author: Nicolas THAIZE
        """
        if auto_ordinal_encode:
            self.set_df(self.ordinal_encoding(self.df))

        am = np.mean(self.df, axis=0)
        gm = np.power(np.prod(self.df, axis=0), 1 / self.df.shape[0])
        disp_ratio = am/gm
        disp_ratio = disp_ratio[disp_ratio >= threshold].sort_values(ascending=False)
        
        fig, ax = plt.subplots()
        plt.bar(disp_ratio.index, disp_ratio)
        plt.yscale(plot_y_scale)
        plt.xticks(rotation = 45)
        fig.subplots_adjust(bottom=0.3)
        plt.title('Dispertion ratio value (non +infinity) by column')
        plt.ylabel('Dispertion ratio value (non +infinity)')
        plt.xlabel('Column name')
        plt.show()

    def drop_columns_by_dispertion_ratio_value(self, auto_ordinal_encode=True, threshold=100):
        """Plot dispertion ratio calculus on whole dataset 
        Args:
            auto_ordinal_encode (boolean, optional): If true, encode categorical features
                By default True
            threshold (integer, optional): min value on dispertion ratio to be kept
                By default 100
        @Author: Nicolas THAIZE
        """

        if auto_ordinal_encode:
            self.set_df(self.ordinal_encoding(self.df))

        am = np.mean(self.df, axis=0)
        gm = np.power(np.prod(self.df, axis=0), 1 / self.df.shape[0])
        disp_ratio = am/gm
        disp_ratio = disp_ratio[disp_ratio >= threshold].sort_values(ascending=False)

        self.set_df(self.df.loc[: , disp_ratio.index])        

    #def normalized_cut_score_by_feat(self, x='auto', auto_ordinal_encode=True):
    #    """Calculate the score for each feat using NormalizedCut model
    #    Args:
    #        x (integer): Number of features to keep
    #        auto_ordinal_encode (boolean, optional): If true, encode categorical features
    #            By default True
    #    Returns:
    #        normalized_cut fitted model
    #    @Author: Nicolas THAIZE
    #    """
    #    if auto_ordinal_encode:
    #        self.set_df(self.ordinal_encoding(self.df))
    #    self.set_df(self.df.dropna(axis=1, how='any'))

    #    if x=="auto":
    #        x=len(self.df.columns)
    #    response = NormalizedCut(x)._calc_scores(self.df)
    #    return response

    #def normalized_cut_fit(self, x, prep_data=False):
    #    """Fit a normalized cut model
    #    Args:
    #        x (integer): Number of features to keep
    #        prep_data (boolean, optionnal): simple data cleaning
    #            By default False
    #    Returns:
    #        normalized_cut fitted model
    #    @Author: Nicolas THAIZE
    #    """
    #    if prep_data:
    #        self.prep_data()
    #    return NormalizedCut(x).fit(self.df)

    #def normalized_cut_transform(self, normalized_cut, prep_data=False):
    #    """Transform a dataset using a fitted normalized cut model
    #    Args:
    #        normalized_cut (NormalizedCut): Fitted normalized cut model
    #        prep_data (boolean, optionnal): simple data cleaning
    #            By default False
    #    Returns:
    #        ndarray of transformed data
    #    @Author: Nicolas THAIZE
    #    """
    #    if prep_data:
    #        self.prep_data()
    #    return normalized_cut.transform(self.df)

    #def prep_data(self):
    #    """Really simple data preparation to prevent errors during functions tests
    #    Args:
    #    Returns:
    #        filtered dataset
    #    @Author: Nicolas THAIZE
    #    """
    #    self.set_df(self.ordinal_encoding(self.df))
    #    self.set_df(self.df.dropna(axis=1, how='any'))

if __name__ == "__main__":
    df = get_data(file_path = "../data/en.openfoodfacts.org.products.csv", nrows=50)
    
    
    
    # Normalized cut example
    #feature_selection = FeatureSelection(df)
    #print(feature_selection.normalized_cut_score_by_feat())
    
    # Dispertion ratio example
    feature_selection = FeatureSelection(df)
    feature_selection.plot_dispertion_ratio()
    feature_selection.drop_columns_by_dispertion_ratio_value()

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
    #unittest.main()
