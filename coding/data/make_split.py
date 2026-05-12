from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit as MSSS
import polars as pl


def make_split(df, target_columns, target_type_columns):

    mlb = MultiLabelBinarizer(sparse_output=True)  # sparse → memory-friendly
    Y = mlb.fit_transform(label_sets)


subject_labels = defaultdict(set)

for row in df.itertuples(index=False):
    subject_labels[row.subject_id].update(row.target)

subjects, label_sets = zip(*subject_labels.items())  # len = #subjects

mlb = MultiLabelBinarizer(sparse_output=True)  # sparse → memory-friendly
Y = mlb.fit_transform(label_sets)
