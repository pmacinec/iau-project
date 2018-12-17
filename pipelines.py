#######################################
#             IAU PROJECT             #
# Authors: Peter Macinec, Lukas Janik #
#     Class files for pipelines       #
#######################################

# import libraries
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
import re
from functools import reduce
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

class ColumnsSelector(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None, **fit_params):
        return self
    
    def transform(self, df, **transform_params):
        return df[self.columns]


class ColumnsEncoder(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.categories = {}
        
    def fit(self, df, y=None, **fit_params):
        for col in self.columns:
            self.categories[col] = set([str(col) + '_' + str(x) for x in df[col].unique()])

        return self

    def dummies(self, df, column):
        return pd.concat([df, pd.get_dummies(df[column],prefix = column)], axis='columns')
    
    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for col in self.columns:
            categories = set([str(col) + '_' + str(x) for x in df[col].unique()])
            to_drop = [x for x in categories if x not in self.categories[col]]
            to_add = [x for x in self.categories[col] if x not in categories]
            
            df_copy = self.dummies(df_copy, col)
            df_copy.drop(to_drop, axis=1, inplace=True)
            if len(to_add):
                df_copy = pd.concat([df_copy, pd.DataFrame(np.zeros((len(df_copy), len(to_add))), columns=to_add)], axis='columns')
            df_copy.drop(col, axis=1, inplace=True)

        return df_copy


class MergeRemoveDuplicates(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df, y=None, **fit_params):
        return self

    def merge(self, duplicates):
        return reduce(lambda x,y: x if not pd.isna(x) and not str(x).startswith('?') else y, duplicates)

    def transform(self, df, **transform_params):
        df_copy = df.copy()

        deduplicated = df_copy[df_copy.duplicated(subset=['name', 'address'], keep=False)].groupby(['name','address']).agg(self.merge).reset_index()
        df_copy.drop_duplicates(subset=['name','address'], keep=False, inplace=True)
        df_copy = df_copy.append(deduplicated)
        return df_copy.reset_index()


class DropRowsNanColumn(TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        df = df[pd.notnull(df[self.column])]
        return df


class NanUnifier(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for column in self.columns:
            df_copy[column] = df_copy[column].apply(lambda x: np.NaN if '?' in str(x) else x)
            
        return df_copy


class BooleanUnifier(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for column in self.columns:
            df_copy[column] = df_copy[column].map(lambda x: str(x).lower().startswith('t'), na_action='ignore')
        return df_copy

class WordsUnifier(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None, **fit_params):
        return self
    
    def unify_words(x):
        x = x.lower().strip()
        return x.replace('_','-')

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for column in self.columns:
            df_copy[column] = df_copy[column].map(lambda x: unify_words(x), na_action='ignore')
        return df_copy


class DateFormatUnifier(TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, df, y=None, **fit_params):
        return self

    def fix_date(self, x):
        x = str(x)
        to_return = ''
        x = str(x)
        p_year = re.compile('(^[^-/]+)')
        p_month = re.compile('[-/](.*)[-/]')
        p_day = re.compile('[-/][0-9]*[-/]([0-9]*)')

        year = p_year.findall(x)[0]
        day = p_day.findall(x)[0]
        month = p_month.findall(x)[0]

        swap = False

        if int(day) > 31:
            tmp = day
            day = year
            year = tmp
            swap = True
        if int(month) > 12:
            tmp = month
            month = year
            year = month
            swap = True
        if not swap and len(str(year)) <= 3:
            year = '19' + year

        to_return = str(day) + '/' + str(month) + '/' + str(year)

        return to_return

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        df_copy[self.column] = df_copy[self.column].map(lambda x: self.fix_date(x), na_action='ignore')
        return df_copy


class ClassUnifier(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        df['class_status'] = df['class'].str.extract('(^[^.]+)', expand=False).str.strip().str.lower()
        return df


class DropColumn(TransformerMixin):
    def __init__(self, column):
        self.column = column
        
    def fit(self, df, y=None, **fit_params):
        return self
    
    def transform(self, df, **transform_params):
        df = df.drop([self.column], axis=1)
        return df


class ColumnExpander(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        df['bred'] = df['personal_info'].str.extract('(^[^|]+)', expand=False).str.strip().str.lower()
        df['origin'] = df['personal_info'].str.extract('[|](.*)\r', expand=False).str.strip().str.lower()
        df['study'] = df['personal_info'].str.extract('[\n](.*)--', expand=False).str.strip().str.lower()
        df['status1'] = df['personal_info'].str.extract('--(.*)[|]', expand=False).str.strip().str.lower()
        df['status2'] = df['personal_info'].str.extract('--.*[|](.*)', expand=False).str.strip().str.lower()
        return df

class ColumnToNumber(TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        df[self.column] = pd.to_numeric(df[self.column])
        return df


class MeasuredValuesFixer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None, **fit_params):
        return self
    
    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for column in self.columns:
            if len(df_copy[df_copy[str(column) + ' measured'].isnull() & ~df_copy[column].isnull()]):
                    df_copy.loc[df_copy[str(column) + ' measured'].isnull() & ~df_copy[column].isnull(), str(column) + ' measured'] = True
            if len(df_copy[df_copy[str(column) +' measured'].isnull() & df_copy[column].isnull()]):
                    df_copy.loc[df_copy[str(column) + ' measured'].isnull() & df_copy[column].isnull(), str(column) + ' measured'] = False
        return df_copy


class Normalizer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None, **fit_params):
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for column in self.columns:
            df_copy[column] = np.log(df_copy[column])
        return df_copy


class OutliersRemover(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.Q1 = {}
        self.Q3 = {}

    def fit(self, df, y=None, **fit_params):
        for column in self.columns:
            self.Q1[column] = df[column].quantile(.25)
            self.Q3[column] = df[column].quantile(.75)
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for column in self.columns:
            Q1 = self.Q1[column]
            Q3 = self.Q3[column]
            lower_outlier = Q1 - (Q3 - Q1) * 1.5
            upper_outlier = Q3 + (Q3 - Q1) * 1.5

            df_copy = df_copy[(df_copy[column] >= lower_outlier) & (df_copy[column] <= upper_outlier)]
        return df_copy


class OutliersReplacer(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.Q1 = {}
        self.Q3 = {}
        self.quantile_95 = {}
        self.quantile_05 = {}

    def replace_outlier(self, x, column):
        Q1 = self.Q1[column]
        Q3 = self.Q3[column]
        lower_outlier = Q1 - (Q3 - Q1) * 1.5
        upper_outlier = Q3 + (Q3 - Q1) * 1.5

        if x < lower_outlier:
            return self.quantile_05[column]
        elif x > upper_outlier:
            return self.quantile_95[column]
        else:
            return x

    def fit(self, df, y=None, **fit_params):
        for column in self.columns:
            self.Q1[column] = df[column].quantile(.25)
            self.Q3[column] = df[column].quantile(.75)
            self.quantile_95[column] = df[column].quantile(.95)
            self.quantile_05[column] = df[column].quantile(.05)
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for column in self.columns:
            df_copy[column] = df_copy[column].apply(lambda x: self.replace_outlier(x, column))
            
        return df_copy


class NumMedianFiller(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.medians = {}
        
    def fit(self, df, y=None, **fit_params):
        for column in self.columns:
            self.medians[column] = df[column].median()
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for column in self.columns:
            df_copy[column] = df[column].apply(lambda x: self.medians[column] if pd.isna(x) else x)
        return df_copy


class NumModelFiller(TransformerMixin):
    def __init__(self, column, model):
        self.column = column
        self.model = model
        self.medians = {}

    def fit(self, df, y=None, **fit_params):
        # choose all numeric attributes and drop nan values
        num_only = df.select_dtypes(include=[np.number]).dropna()

        # create X and y training subsets
        X_train = num_only[num_only.columns.difference([self.column])]
        y_train = num_only[self.column]

        self.columns = X_train.columns.values
        for column in self.columns:
            self.medians[column] = X_train[column].median()

        scores = cross_val_score(self.model, X_train, y_train, scoring="neg_mean_squared_error", cv=10)
        print("Score for column " + self.column + ": " + str(scores.mean()))

        self.model.fit(X_train, y_train)
        
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        if df[self.column].isnull().any():
            # choose only those rows that have missing value in 'column' attribute
            empty_rows = df_copy[df_copy[self.column].isnull()][self.columns]

            for column in self.columns:
                empty_rows.loc[empty_rows[column].isnull(), column] = self.medians[column]
            
            # create X subset, that will predict
            X_pred = empty_rows[empty_rows.columns.difference([self.column])]

            # predict new values
            y_pred = self.model.predict(X_pred)

            # fill missing values with predicted ones
            df_copy.loc[empty_rows.index, self.column] = y_pred

        return df_copy


class CategoricalMostFrequentFiller(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.most_frequents = {}
        
    def fit(self, df, y=None, **fit_params):
        for column in self.columns:
            self.most_frequents[column] = str(df[column].mode()[0])
        return self
    
    def transform(self, df, **transform_params):
        df_copy = df.copy()
        for column in self.columns:
            if (type(df_copy[column].values[0]) == bool):
                self.most_frequents[column] = bool(self.most_frequents[column])
            df_copy.loc[df_copy[column].isnull(), column] = self.most_frequents[column]
        return df_copy


class CategoricalModelFiller(TransformerMixin):
    def __init__(self, column, model):
        self.column = column
        self.model = model
        self.most_frequents = {}

    def fit(self, df, y=None, **fit_params):

        # choose all numeric attributes and drop nan values
        num_only = df.select_dtypes(include=[np.number])

        num_only[self.column] = df[self.column]

        num_only = num_only.dropna()

        # create X and y training subsets
        X_train = num_only[num_only.columns.difference([self.column])]
        y_train = num_only[self.column].astype(str)

        self.columns = X_train.columns.values
        for column in self.columns:
            self.most_frequents[column] = str(X_train[column].mode()[0])

        scores = cross_val_score(self.model, X_train, y_train, scoring="accuracy", cv=10)
        print("Score for column " + self.column + ": " + str(scores.mean()))
        
        self.model.fit(X_train, y_train)
        
        return self

    def transform(self, df, **transform_params):
        df_copy = df.copy()
        if df[self.column].isnull().any():
            # choose only those rows that have missing value in 'column' attribute
            empty_rows = df_copy[df_copy[self.column].isnull()][self.columns]

            for column in self.columns:
                empty_rows.loc[empty_rows[column].isnull(), [column]] = self.most_frequents[column]

            # create X subset, that will predict
            X_pred = empty_rows

            # predict new values
            y_pred = self.model.predict(X_pred)
            
            # fill missing values with predicted ones
            df_copy.loc[empty_rows.index, self.column] = y_pred

            df_copy[self.column] = df_copy[self.column].astype(str)
            
        return df_copy