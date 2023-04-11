import pandas as pd
import numpy as np

def split_ean13(df, drop_ean13=False):
    """
    This function split code column into 3 separate columns if it is digit, its length is 13 and don't start with 200
    because Open Food Facts assigns a number starting with the 200 reserved prefix for products without a barcode.
    Code column is barcode of the product (can be EAN-13 or internal codes for some food stores).
    Position 1 to 3 : country code
    Position 3 to 7 : manufacturer code
    Position 8 to 12 : product code
    Position 13 : check digit that will be removed
    Parameters:
        df : pandas dataframe (The input dataframe containing the EAN13 codes in a column named "code")
        drop_ean13 : bool (Default=False), drop code column if set True
    Returns:
        df : pandas dataframe with the separated EAN13 columns
    Author:
        Joëlle Sabourdy
    """
    # Check if code column exists
    if 'code' not in df.columns:
        raise ValueError('code column not found in dataframe.')
    
    # Create new columns
    df['EAN13_country'] = np.nan
    df['EAN13_company'] = np.nan
    df['EAN13_product'] = np.nan
    
    # Loop over rows
    for i, row in df.iterrows():
        ean = str(row['code'])
        # Check if code is valid with EAN13
        if not ean.isdigit():
            continue
        # Check code length
        if len(ean) != 13:
            continue
        # Check if code starts with 200
        if ean.startswith('200'):
            continue
        # Split EAN13
        prefix = ean[:3]
        company = ean[2:7]
        product = ean[7:12]
        # Store values
        df.at[i, 'EAN13_country'] = prefix
        df.at[i, 'EAN13_company'] = company
        df.at[i, 'EAN13_product'] = product
    
    # Drop EAN13 column if specified
    if drop_ean13:
        df.drop('code', axis=1, inplace=True)
    
    return df



if __name__ == "__main__":
    # Consider dataset containing ramen product
    df = pd.DataFrame({
        'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Tanoshi', 'Cup Noodles'],
        'style': ['cup', 'cup', 'cup', 'pack', 'pack', 'cup'],
        'code': [1234567890123, '123456', '2007890123456', '', '0067890123456', '1234567890az3456']
        })
    # Split code column
    df = split_ean13(df, drop_ean13=False)
    print(df)
