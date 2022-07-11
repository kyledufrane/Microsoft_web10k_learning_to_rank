import modin.pandas as pd

# Preprocess and evaluate the dataset

def preprocess_features():
    '''This function processes and creates our feature columns descriptions'''
    # Read in the features file
    features = pd.read_csv('features.csv')
    # Create new header and replace spaces with underscore
    new_header = features.iloc[0].str.replace(' ','_')
    # Remove the first row which is now the new header
    features = features[1:]
    # Set new headers
    features.columns = new_header
    # Only the first cell for each category is filled. Using forward will
    # will allow me to map each category to their sub-categories located
    # in the stream column 
    features['feature_description'] = features['feature_description'].ffill()
    # Replacing characters to allign with TensorFlows regex requirements
    character_removal = [' ', '(', ')', '*']
    for char in character_removal:
        features['feature_description'] = features['feature_description'].str.replace(char, '_')
        features['stream'] = features['stream'].astype(str).str.replace(char, '_')
    # Setting column type to string for mapping within the load_rename_save function
    features['feature_id'] = features['feature_id'].astype(str)
    # Creating new column to map features to existing dataset
    features['cols'] = 'string'
    # Looping over all features and creating new column name
    for idx in range(len(features)):
        if str(features.iloc[idx]['stream']) != 'nan':
            features['cols'].iloc[idx] = features['feature_description'].iloc[idx] + '_' + features['stream'].iloc[idx]
        else:
            features['cols'].iloc[idx] = features['feature_description'].iloc[idx]
    return features

# ==================================================================================================================================

def label_columns(df):
    '''This function labels the columns by descriptions
       found on the microsoft research page'''    
        
    for col in df.columns:
        if col == 0:
            df.rename({col : 'relevance_label'}, axis=1, inplace=True)
        elif col == 1:
            df.rename({col : 'query_id'}, axis=1, inplace=True)
        else:
            df.rename({col : f'feature_{col - 1}'}, axis=1, inplace=True)
            
    return df

# ==================================================================================================================================

def load_rename_save(folder_num):
    '''This function reads in all data located in folder n,
       labels the columns, removes uneeded elements from the cells (i.e. 'qid:1' the qid is uneeded),
       and saves the files as a parquet within folder n'''
    
    for folder in folder_num:
        # Load data
        df_train = pd.read_csv(f'Fold{folder}/train.txt', sep=' ', header=None)
        df_test = pd.read_csv(f'Fold{folder}/test.txt', sep=' ', header=None)
        df_val = pd.read_csv(f'Fold{folder}/vali.txt', sep=' ', header=None)
        
        # Label the columns
        df_train = label_columns(df_train)
        df_test = label_columns(df_test)
        df_val = label_columns(df_val)
        
        # Remove 'n:' from each column. The dataset assigned each feature number
        # to the cells value which needs to be removed to get the data into int/float format
        dataframes = {'train': df_train, 'test': df_test, 'val': df_val}
        for k, df in dataframes.items():
            for i in range(1,len(df.columns)-1):
                df[f'feature_{i}'].replace(f'{i}:', '', regex=True, inplace=True)          
            
        # Only query_id was different than all of the other columns when assigning 
        # the prefix to the values. Here we remove 'qid:' from each cell
            df['query_id'].replace('qid:', '', regex=True, inplace=True)

        # Rename the feature columns from the given descriptions on Microsofts webiste   
        features = preprocess_features()
        
        for k, df in dataframes.items():
            for idx in range(len(features)):
                id_ = features.iloc[idx]['feature_id']
                for col in df.columns:
                    if str(id_) == col.lstrip('feature_'):
                        df.rename({col: features.iloc[idx]['cols']}, axis=1, inplace=True)
        
        # Save the cleaned dataset as a csv
        df_train.to_csv(f'Fold{folder}/df_train.csv', index=False)
        df_test.to_csv(f'Fold{folder}/df_test.csv', index=False)
        df_val.to_csv(f'Fold{folder}/df_val.csv', index=False)

# ==================================================================================================================================        
        
def data_stats(folder_num):
    ''' This function is to collect basic stats from the dataset. '''
    for folder in folder_num:
        # Load the data
        df_train = pd.read_csv(f'Fold{folder}/df_train.csv')
        df_test = pd.read_csv(f'Fold{folder}/df_test.csv')
        df_val = pd.read_csv(f'Fold{folder}/df_val.csv')
        
        # Collect metrics for below stats
        len_train = len(df_train)
        len_test = len(df_test)
        len_val = len(df_val)
        total = len_train + len_test + len_val
        
        # Print length of all datasets and the overal balance between the splits
        print('*'*24 + ' ' + f'Folder Number {folder}' + ' ' + '*'*24)
        print(f'Total rows in training set {folder}: {len_train}')
        print(f'Total rows in testing set {folder}: {len_test}')
        print(f'Total rows in validation set {folder}: {len_val}')
        print('='*64)
        print(f'The training set contains {round(len_train/total, 2) * 100}% of the total data')
        print(f'The testing set contains {round(len_test/total, 2) * 100}% of the total data')
        print(f'The validation set contains {round(len_val/total, 2) * 100}% of the total data')
        print('='*64)
        
        # Create new dataframe showing NaN values
        df_train_ = pd.DataFrame(df_train.isna().sum(), columns=['NaN_values'])
        df_test_ = pd.DataFrame(df_test.isna().sum(), columns=['NaN_values'])
        df_val_ = pd.DataFrame(df_val.isna().sum(), columns=['NaN_values'])
        
        # Mapping of NaN dataframes
        nan_dataframes = {'df_train':df_train_, 'df_test':df_test_, 'df_val':df_val_}
        
        # Print the total percentage of missing values per column in each dataframe
        for k,df in nan_dataframes.items():
            df = df[df['NaN_values'] > 0]
            total_missing = [len(df.index)]
            for missing in total_missing:
                if k == 'df_train':
                    print(f'df_train_ Column {df.index[missing - 1]} is missing {df.values[missing - 1] / len(df_train)}% of its data')
                elif k == 'df_test':
                    print(f'df_test_ Column {df.index[missing - 1]} is missing {df.values[missing - 1] / len(df_test)}% of its data')
                else:
                    print(f'df_val_ Column {df.index[missing - 1]} is missing {df.values[missing - 1] / len(df_val)}% of its data')
        
        # Mapping of initial dataframes
        dataframes = {'df_train': df_train, 'df_test': df_test, 'df_val': df_val}
        
        # Calculating the distribution of the relevance column
        for k,df in dataframes.items():
            df_len = len(df)
            relevance_counts = df['relevance_label'].value_counts()
            print('='*64)
            print('*'*16 + ' ' + f'{k} Relevance Class Balance' + ' ' + '*'*16)
            for i in [0,1,2,3,4]:
                print(f'Rank {i}: Total Count: {relevance_counts[i]} Percentage: {round(relevance_counts[i]/df_len,2) * 100}%')
        print(' ')
        
# ==================================================================================================================================        
        
def drop_unwanted_cols(folder_num):
    '''This function drops column 137 from each dataframe due to 137 missing 
        100% of its values across all datasets'''
    for folder in folder_num:
        df_train = pd.read_csv(f'Fold{folder}/df_train.csv')
        df_test = pd.read_csv(f'Fold{folder}/df_test.csv')
        df_val = pd.read_csv(f'Fold{folder}/df_val.csv')
        
        df_train.drop('feature_137', axis=1, inplace=True)
        df_test.drop('feature_137', axis=1, inplace=True)
        df_val.drop('feature_137', axis=1, inplace=True)
        
        df_train.to_csv(f'Fold{folder}/df_train.csv', index=False)
        df_test.to_csv(f'Fold{folder}/df_test.csv', index=False)
        df_val.to_csv(f'Fold{folder}/df_val.csv', index=False)
        print(f'Finished Cleaning Fold{folder}')

# ==================================================================================================================================

def build_complete_dataset(folder_num):
    '''This function takes in a folder number related to Fold[folder_number]
       and builds a complete datasets across the train/test/val subsets'''

    train_df = pd.read_csv(f'Fold{folder_num}/df_train.csv')
    test_df = pd.read_csv(f'Fold{folder_num}/df_test.csv')
    val_df = pd.read_csv(f'Fold{folder_num}/df_val.csv')

    df = pd.concat([train_df, test_df, val_df], axis=0)

    return df
    