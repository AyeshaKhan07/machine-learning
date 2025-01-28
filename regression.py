import pandas as pd

dataset = pd.read_excel("HousePricePrediction.xlsx")

# getting only the columns we need to train the model
dataset = dataset[['LotArea', 'OverallCond', 'YearBuilt', 'SalePrice']]
print("head", dataset.head())

obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
# print("Categorical variables:",len(object_cols))

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
# print("Integer variables:",len(num_cols))

fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)
# print("Float variables:",len(fl_cols))
