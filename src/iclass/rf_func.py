import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

def feature_importance(feature_names: pd.Index, clf: RandomForestClassifier) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    feature_names : pd.Index
        Name of the columns of the dataframe used to train the RF.
    clf : RandomForestClassifier
        Trained RF for which the importance of features shall be checked.

    Returns
    -------
    pd.DataFrame
        Ranked importance of the features in the random forest classifier based on the Gini index.
    """

    importances = clf.feature_importances_
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    
    return feature_importances

def train_rf(df_train: pd.DataFrame, config: dict=None) -> RandomForestClassifier:
    """
    Train a Random Forest Regressor for the classification of irf classes.

    Parameters
    ----------
    train: `pandas.DataFrame`
    config: dictionary

    Returns
    -------
    The trained classifier object.
    """
     
    model = RandomForestClassifier
    logger.info(f"Number of events for training: {df_train.shape[0]}")

    if config:
        classifier_args = config['random_forest_args']
        features = config['random_forest_features']
        clf = model(**classifier_args)
        
        logger.info(f"Given features: {features}")
        logger.info("Training Random Forest Classifier for PSF Classes ...")
        
        clf.fit(df_train[features],
                df_train['psf_class'])
        
    else:
        clf = model()
        
        logger.info("No features provided, use all columns.")
        logger.info("No RF settings provided, use scikit-learn defaults.")
        logger.info("Training Random Forest Calssifier for PSF Classes...")
        
        clf.fit(df_train.drop(columns=['psf_class']),
                df_train['psf_class'])

    logger.info("Model {} trained!".format(model))
    return clf