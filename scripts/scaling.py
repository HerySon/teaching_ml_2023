from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler 
#define functions used
def scaling_data(df, scaler_name):
   # la fonction permet à l'utilsateur de choisir sa methode ( MinMaxScaler, StandardScaler,StandardScaler) pour faire le scaling du data framme

# scaler_name : la methode à utilisée 
# ctake only variables we will experimen
df = df[features]
#split data
X_train, X_test, y_train, y_test = train_test_split(df[features],test_size=0.3,random_state=0)
# choix de la methode 
if scaler_name='StandardScaler' :
   #fit the scaler to the train set 
scaler_std = StandardScaler().fit(X_train)

# transform data
X_train_scaled_std = scaler_std.transform(X_train)
else if scaler_name ='MinMaxScaler' :
 # fit the scaler to the train set 
scaler_minmax = MinMaxScaler().fit(X_train)

# transform data
X_train_scaled_minmax = scaler_minmax.transform(X_train)
else if scaler_name='MaxAbsScaler' :
scaler_maxabs = MaxAbsScaler().fit(X_train)
 else if scaler_name='RobustScaler' :
scaler_rbs = RobustScaler().fit(X_train)
X_train_scaled_rbs = scaler_rbs.transform(X_train)
else : 
 print('saisir une methode')
return rien 



