import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


def random_imputation(df, feature):

        number_missing = df[feature].isnull().sum()
        observed_values = df.loc[df[feature].notnull(), feature]
        df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)

        return df
def det_regression(df):

        #Random imp first:
        missing_columns = df.columns
        for feature in missing_columns:
            df[feature + '_imp'] = df[feature]
            df = random_imputation(df, feature)
        df_random=df.copy()


        deter_data = pd.DataFrame(columns = ["Det" + name for name in missing_columns])
        for feature in missing_columns:

            deter_data["Det" + feature] = df[feature + "_imp"]
            parameters = list(set(df.columns) - set(missing_columns) - {feature + '_imp'})
            #Create a Linear Regression model to estimate the missing data

            X = df[parameters].values.tolist()
            y = df[feature + '_imp'].values.tolist()
            poly = PolynomialFeatures(degree=2)
            X_ = poly.fit_transform(X)

            predict_ = poly.fit_transform(df[parameters])[df[feature].isnull()]

            # print(parameters)

            model = linear_model.LinearRegression()
            model.fit(X = X_, y = y)
        #     model.coef_[model.coef_<0.00001]=0
            #observe that I preserve the index of the missing data from the original dataframe
            deter_data.loc[df[feature].isnull(), "Det" + feature] = model.predict(predict_)


        deter_data.columns=[x.split('Det')[1] for x in deter_data.columns]

        return  deter_data,df_random