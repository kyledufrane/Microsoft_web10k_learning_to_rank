import os
import modin.pandas as pd
import numpy as np

import tensorflow_ranking as tfr
import tensorflow as tf

from helpers import preprocess_features

def build_model(model_dir,
                train_input_pattern:str='tfrecord_1/train.tfrecords',
                valid_input_pattern:str='tfrecord_1/val.tfrecords',
                num_epochs:int=5,
                optimizer:str='adagrad',
                loss:str="approx_ndcg_loss",
                hidden_layer_dims:list=[1024,512,256],
                activation=tf.nn.relu,
                use_batch_norm:bool=True,
                batch_norm_moment:float=0.999,
                dropout:float=0.5,
                list_size:int=200,
                preprocessor:bool=False,
                train_batch_size:int=128,
                valid_batch_size:int=128,
                steps_per_epoch:int=5000,
                validation_steps:int=125,
                learning_rate:float=0.05):
    
    '''Trains and validates data via TensorFlowRanking Pipeline with DNNScorer
    
    Args:
        model_dir: Directory location to save modeling information (Note: Code will generate new folder)
        train_input_pattern: training data file location
        valid_input_pattern: validation data file location
        optimizer: tf.keras.optimizers
        num_epochs: total training epochs
        loss: model loss function --> default: 'approx_ndcg_loss'
        hidden_layer_dims: hidden layer dimensions
        activation: activation for each layer
        use_batch_norm: use of batch normalization for each layer
        batch_norm_moment: momentum for the moving average in batch normalization
        dropout: dropout rate
        preprocessor: use of preprosser (Note: Current implmentation = log1p)
        train_batch_size: training batch size
        valid_batch_size: validation batch size
        steps_per_epoch: number of training steps per epoch
        validation_steps: number of validation steps per epoch
        learning_rate: learning rate     
    '''
    
    # Define Scorer
    scorer = tfr.keras.model.DNNScorer(hidden_layer_dims=hidden_layer_dims,
                                        output_units=1,
                                        activation=activation,
                                        use_batch_norm=use_batch_norm,
                                        batch_norm_moment=batch_norm_moment,
                                        dropout=dropout)

    # Collect feature names for mapping
    features = preprocess_features()
    feature_cols = np.array(features['cols'])
    
    # Create specs for pipeline
    context_spec_ = {}
    example_spec_ = {feat: tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=0.0) for feat in feature_cols}
    label_spec_ = ('relevance_label', tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=-1))

    # Log1p preprocessing spec for data normalization
    preprocess_spec = {
        **{name: lambda t: tf.math.log1p(t * tf.sign(t)) * tf.sign(t)
           for name in example_spec_.keys()}
    }

    # Define Input Creator
    input_creator= tfr.keras.model.FeatureSpecInputCreator(
            context_feature_spec={},
            example_feature_spec=example_spec_)
    
    # Define Model
    if preprocessor == False:
        model_builder = tfr.keras.model.ModelBuilder(
            input_creator=input_creator,
            preprocessor=tfr.keras.model.PreprocessorWithSpec(),
            scorer=scorer,
            mask_feature_name="example_list_mask",
            name="model_builder")
    else:
        model_builder = tfr.keras.model.ModelBuilder(
        input_creator=input_creator,
        preprocessor=tfr.keras.model.PreprocessorWithSpec(preprocess_spec),
        scorer=scorer,
        mask_feature_name="example_list_mask",
        name="model_builder")

    # Define Dataset Parameters
    dataset_hparams = tfr.keras.pipeline.DatasetHparams(
        train_input_pattern=train_input_pattern,
        valid_input_pattern=valid_input_pattern,
        train_batch_size=train_batch_size,
        valid_batch_size=valid_batch_size,
        list_size=list_size,
        dataset_reader=tfr.keras.pipeline.DatasetHparams.dataset_reader)

    # Define Dataset Builder
    dataset_builder = tfr.keras.pipeline.SimpleDatasetBuilder(
        {},
        example_spec_,
        mask_feature_name="example_list_mask",
        label_spec=label_spec_,
        hparams=dataset_hparams,
        sample_weight_spec=None)

    # Define Pipeline Hparams
    pipeline_hparams = tfr.keras.pipeline.PipelineHparams(
        model_dir=model_dir,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        learning_rate=learning_rate,
        loss=loss,
        optimizer=optimizer)
    
    # Define Ranking Pipeline
    ranking_pipeline = tfr.keras.pipeline.SimplePipeline(
        model_builder,
        dataset_builder=dataset_builder,
        hparams=pipeline_hparams)
    
    # Train and Validate
    ranking_pipeline.train_and_validate(verbose=1)
