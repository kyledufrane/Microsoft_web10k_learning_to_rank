
# Import needed libraries
import os
import numpy as np

import modin.pandas as pd

import tensorflow as tf
from tensorflow_serving.apis import input_pb2

from sklearn.datasets import load_svmlight_file

from helpers import preprocess_features

class SVM_to_TFRecord():
    '''SVM_to_TFRecord converts .txt files to TFRecords using sklearns load_svmlight_file which 
       loads the files in SVMlight format allowing faster computations. Loading .txt with this
       method returns three arrays when query_id is set to True. 
       
           Array 0 = Features
           Array 1 = Relevance
           Array 2 = Query ID
           
      SVM_to_TFRecord Args:
                          data_path: Path to load .txt data
                          tfrecord_path: Path to save .tfrecords           
           '''
    
    def __init__(self, data_path:str='', tfrecord_path:str=''):
        self.data_path = data_path
        self.tfrecord_path = tfrecord_path
        
        if not os.path.isdir(self.data_path):
            print('Please enter correct data path!!!!')

        if not os.path.isdir(self.tfrecord_path):
            os.makedirs(self.tfrecord_path)

    # Helper functions for creating TF Features
    # Note: There are only two feature types in 
    # the datasets
    def _float_feature(self, value):
      """Returns a float_list from a float / double."""
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
      """Returns an int64_list from a bool / enum / int / uint."""
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def create_tfrecord(self, data, tfrecord_path):
        
        options_ = tf.io.TFRecordOptions(compression_type='GZIP')
        with tf.io.TFRecordWriter(tfrecord_path) as writer:

            # Create Example list 
            elwc = input_pb2.ExampleListWithContext()
            # Save the last query id for filtering
            last_query_id = None
            # Map Feature names to dictionary
            features_ = preprocess_features()
            # Extract column names
            labels = np.array(features_['cols'])

            for row in range(data.shape[0]):
                # Select data from each row
                relevance_label, query_id, features = data[row,0], data[row,1], data[row,2:]
                # Create Example Dict
                example_dict = {
                    f'{feat_name}':self._float_feature(feat_val)
                    for feat_name, feat_val in zip(labels, features)
                }
                example_dict['relevance_label'] = self._int64_feature(int(relevance_label))
                # Create Features
                example_ = tf.train.Example(features=tf.train.Features(feature=example_dict))
                # Create ELWC by query id
                if query_id != last_query_id:
                    if last_query_id != None:
                        writer.write(elwc.SerializeToString())
                    last_query_id = query_id
                    elwc = input_pb2.ExampleListWithContext()
                    elwc.examples.append(example_)
                else:
                    elwc.examples.append(example_)
            # Writing the final query
            writer.write(elwc.SerializeToString())
            
    def load_and_convert_data(self):
        
        train = load_svmlight_file(f'{self.data_path}/train.txt', query_id=True)
        test = load_svmlight_file(f'{self.data_path}/test.txt', query_id=True)
        val = load_svmlight_file(f'{self.data_path}/vali.txt', query_id=True)
        
        # Note: train, test, val each return three numpy arrays:
            # Array 0 = features
            # Array 1 = relevance
            # Array 3 = Query IDences in a batch fit a given 
            
        data_ = {'train':train, 'test':test, 'val':val}
        
        # Concatenate data into one array
        for label, data in data_.items():
            if label == 'train':
                train = np.concatenate((np.expand_dims(train[1], axis=1), np.expand_dims(train[2], axis=1), train[0].toarray()), axis=1)
                tfrecord_path = f'{self.tfrecord_path}/train.tfrecords'
                self.create_tfrecord(train, tfrecord_path)
                
            elif label == 'test':
                test = np.concatenate((np.expand_dims(test[1], axis=1), np.expand_dims(test[2], axis=1), test[0].toarray()), axis=1)
                tfrecord_path = f'{self.tfrecord_path}/test.tfrecords'
                self.create_tfrecord(test, tfrecord_path)

            else:
                val = np.concatenate((np.expand_dims(val[1], axis=1), np.expand_dims(val[2], axis=1), val[0].toarray()), axis=1)
                tfrecord_path = f'{self.tfrecord_path}/val.tfrecords'
                self.create_tfrecord(val, tfrecord_path)
