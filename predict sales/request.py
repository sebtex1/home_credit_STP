import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'NAME_CONTRACT_TYPE':'Cash loans',
    'CODE_GENDER':'F',
    'FLAG_OWN_CAR':'N',
    'FLAG_OWN_REALTY':'N',
    'CNT_CHILDREN':0,
    'AMT_INCOME_TOTAL':270000.0,
    'AMT_CREDIT':1293502.5,
    'NAME_TYPE_SUITE':'Family',
    'NAME_INCOME_TYPE':'State servant',
    'NAME_EDUCATION_TYPE':'Higher education',
    'NAME_FAMILY_STATUS':'Married',
    'NAME_HOUSING_TYPE':'House / apartment'})

print(r.json())