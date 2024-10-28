"""Script to train a random forest classifier for IRF event classes 
(so far PSFR only). Part of the lst-irf-classes module.
"""
import argparse
import json
import logging

import joblib
import pandas as pd

from iclass.rf_func import feature_importance, train_rf


logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s : %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=r"""
        Random forest training to determine IRF classes for CTAO telescopes.
        """
    )

    parser.add_argument(
        '-i',
        "--input",
        default='',
        help='input Monte Carlo file name'
    )
    parser.add_argument(
        '-p',
        "--prefix",
        default='',
        help='output file name prefix.'
        ' If empty (default) no classifier is written to disc.'
        ' It will be appended with "ic_rf.pkl,"'
        ' when generating the output files'
    )
    parser.add_argument(
        '-e',
        "--event-key",
        default='/dl2/event/telescope/parameters/LST_LSTCam',
        help='input HDF5 file key to read the events from'
    )
    parser.add_argument(
        '-c',
        "--config",
        default='',
        help='Configuration file for RF training.'
    )
    parser.add_argument(
        '-z',
        "--complevel",
        type=int,
        default=7,
        help='scikit-learn data compression level'
    )

    args = parser.parse_args()

    try:
        train_df = pd.read_hdf(args.input, key=args.event_key)
    except FileNotFoundError:
        logger.error("Error: The file %s was not found.", args.input)
    except OSError as e:
        logger.error("Error: An issue occurred while reading the HDF5 file:"
                     "%s", e)
    except json.JSONDecodeError:
        logger.error("Error: Failed to decode JSON from %s.", args.input)

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error("Error: The file %s was not found.", args.config)
    except json.JSONDecodeError:
        logger.error("Error: The file %s is not a valid JSON.", args.config)

    # Train the IRF classes random forest.
    clf = train_rf(train_df, config)

    # Check the most important features of the rf.
    feature_names = train_df[config['random_forest_features']].columns.tolist()
    df_feature_importance = feature_importance(feature_names, clf)

    logger.info("Importance of the features according to their Gini indeces:")
    print(df_feature_importance) 

    # Save the model to a file
    if args.prefix != '':
        logger.info("Saving the RF to '{args.prefix}ic_rf.pkl.pkl'.")
        joblib.dump(clf, f'{args.prefix}ic_rf.pkl.pkl',
                    compress=args.complevel
                    )


if __name__ == "__main__":
    main()
