import tach1_0_anas as tach1_0


import pandas as pd
import numpy as np



class DataFilter:

    def __init__(self) -> None:
        # Initialize DataFilter object, and clean the data by removing rows with more than 30% missing values   
        self.data_cleaned = tach1_0.data_with_less_30_nan(tach1_0.data) 
        pass

 

    # Les autres méthodes restent les mêmes
    def downcast(self,datas: pd.DataFrame, features=None) -> pd.DataFrame:
        """
        Reduces memory usage by downcasting numeric columns in a DataFrame.
        
        Parameters
        ----------
        datas : pd.DataFrame
            Input DataFrame
        features : list, optional
            List of features to downcast, by default None
        
        Returns
        -------
        pd.DataFrame
            DataFrame with downcasted numeric columns
        """
        # Check input types
        print(type(datas))
        assert isinstance(datas, pd.DataFrame), "datas must be a pd.DataFrame"
        assert isinstance(features, list) or features is None, "features must be a list"
        
        # If no features are specified, downcast all columns
        if features is None:
            features = datas.columns
            
        # Loop through features
        for feature in features:
            if datas[feature].dtype == "int64":
                datas[feature] = pd.to_numeric(datas[feature], downcast="integer")
            elif datas[feature].dtype == "float64":
                datas[feature] = pd.to_numeric(datas[feature], downcast="float")
            elif datas[feature].dtype == "object":
                if datas[feature].apply(type).nunique() > 1:
                    continue
                datas[feature] = datas[feature].astype("category")
                
        return datas

 

    def filter_numerical(self, datas: pd.DataFrame, nan_percent: int = 0) -> pd.DataFrame:
        """
        Filter numerical columns in a DataFrame based on the percentage of missing values.

        Parameters
        ----------
        datas : pd.DataFrame
            Input DataFrame
        nan_percent : int, optional
            Percentage of missing values threshold, by default 0

        Returns
        -------
        pd.DataFrame
            DataFrame with filtered numerical columns
        """
            
        features = datas.select_dtypes(include=["integer", "float"]).columns

        result_df = datas[features]

        nans_to_select = result_df.isna().sum()[result_df.isna().sum() / result_df.shape[0] * 100 > nan_percent]

        return result_df.drop(list(nans_to_select.index), axis=1)

 

    def filter_ordinal(self, datas: pd.DataFrame,ordinal_features_names :list) -> pd.DataFrame:
        """
        Filter ordinal columns in a DataFrame based on given column names.

        Parameters
        ----------
        datas : pd.DataFrame
            Input DataFrame
        ordinal_features_names : list
            List of column names of ordinal features to be filtered

        Returns
        -------
        pd.DataFrame
            DataFrame with filtered ordinal columns
        """
        if ordinal_features_names is None:

            return pd.DataFrame()

        features = ordinal_features_names

        return datas[features]

 

    def filter_non_ordinal(self, datas: pd.DataFrame, category_count: int = 100) -> pd.DataFrame:

        features = datas.select_dtypes(include=["category"]).columns

        for feature in features:

            if datas[feature].cat.ordered or datas[feature].nunique() > category_count:

                features = features.drop(feature)

        return datas[features]

 

    def filter_categorical(self, datas: pd.DataFrame) -> pd.DataFrame:
        features = datas.select_dtypes(include=["category"]).columns
        return datas[features]

 

if __name__ == "__main__":

    filter = DataFilter()

    
    print(filter.data_cleaned.dtypes)
    
    datas_DOWNCASTED = filter.downcast(filter.data_cleaned)
    print(datas_DOWNCASTED.dtypes)

    

    print("Setting up filters:")

    print("Numerical columns:")

    print(list(filter.filter_numerical(datas_DOWNCASTED, nan_percent=30).columns))

    print("categorical columns:")

    print(list(filter.filter_categorical(datas_DOWNCASTED).columns))
   