def null__rate_features(df, tx_threshold=50):
    """
        ------------------------------------------------------------------------------
        Goal : 
            - Calculate the null rate of each variable in a dataframe
        ------------------------------------------------------------------------------
        Parameters:
        - df : 
            Dataset to be analyzed
        - tx_threshold : 
            nullity threshold above which a variable is considered to have a high 
            nullity rate (by default 50%)
        -----------------------------------------------------------------------------
        Return :
        - high_null_rate :
            A list of variables with their null rate greater than or equal 
            to the specified threshold,sorted in descending order of null rate
        -----------------------------------------------------------------------------
    """
    
    #Calcul of null rate of each variable
    null_rate = ((df.isnull().sum() / df.shape[0])*100).sort_values(ascending=False).reset_index()
    null_rate.columns = ['Variable','Taux_de_Null']
    
    #Selection of variables with a null rate greater than or equal to the specified threshold
    high_null_rate = null_rate[null_rate.Taux_de_Null >= tx_threshold]
    
    return high_null_rate

def fill_rate_features(df):
    
    """
        ---------------------------------------------------------------------------------
        Goal : 
            - Calculate the fill rate of each variable in a DataFrame
            - Displays a horizontal bar graph that shows the fill rate for each variable.
        ---------------------------------------------------------------------------------
        Parameters:
            - df : Dataframe to be analyzed
        ---------------------------------------------------------------------------------
    """
    
    #Calculate the rate of nullity for each variable 
    filling_features = null__rate_features(df, 0)
    
    #Calculate of fill rate by subtracting the null rate from 100%
    filling_features["fill_rate"] = 100-filling_features["Taux_de_Null"]
    
    #Sort results in descending order of filling rate
    filling_features = filling_features.sort_values("Taux_de_Null", ascending=False)
    
    # Creating the horizontal bar chart with Seaborn
    fig = plt.figure(figsize=(20, 35))
    font_title = {'family': 'serif',
                'color':  '#114b98',
                'weight': 'bold',
                'size': 18, 
                }
    sns.barplot(x="fill_rate", y="Variable", data=filling_features, palette="flare")
    plt.axvline(linewidth=2, color = 'r')
    plt.title("Taux de remplissage des variables dans le jeu de données (%)", fontdict=font_title)
    plt.xlabel("Taux de remplissage (%)")
    
    return plt.show()

def clean_dataframe(df,colonnes_a_garder):
    
    """
        ---------------------------------------------------------------------------------
        Goal : 
            - Clean dataset
        ---------------------------------------------------------------------------------
        Parameters:
            - df : Dataframe to be analyzed
        ---------------------------------------------------------------------------------
    """
    
    fill_rate_features(df)
    
    #Selection of the columns having a suffix '_100g' & adding these columns to the columns already selected
    for column in df.columns:
        if '_100g' in column: colonnes_a_garder.append(column)

    #Delete all columns except those to keep
    colonnes_a_supprimer = [col for col in df.columns if col not in colonnes_a_garder]
    df_garder = df.drop(colonnes_a_supprimer, axis=1)

    """
        Through the plot carried out for the filling rate, we notice
        that there are features that have more than 30 to 40% missing values
    """
    #Calculation of the percentage of missing values in each column
    pourcentages_val_manquantes = df_garder.isnull().mean() * 100

    #Select columns that have more than 40% missing values
    colonnes_val_manquantes = pourcentages_val_manquantes[pourcentages_val_manquantes > 50].index

    #Delete columns that have more than 40% missing values
    df_garder=df_garder.drop(colonnes_val_manquantes,axis=1)

    #Remove duplicates throughout the dataframe
    df_garder.drop_duplicates(inplace=True)

    #List of columns with type float
    colonnes_float = df_garder.select_dtypes(include=['float']).columns

    #Remove inconsistent values
    for col in colonnes_float:
        #Replace negative values with NaN
        df_garder.loc[df_garder[col] < 0, col] = np.nan

    #Save all modifications in a new Csv file
    df_garder.to_csv('dataset_clean.csv', index=False)

    return df_garder

   
if __name__ == "__main__":
    
    colomns_save= ['cities', 'code', 'created_datetime','product_name',
                         'countries_en',"categories","states","pnns_groups_2",
                         "ingredients_text","additives_n","nutriscore_grade","brands"]
    
    clean_dataframe(df,colomns_save)