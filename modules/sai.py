import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time
import pickle
import gc

#################################################################
# Loading Initial data and merging 

def load_initial_files(basepath, save = True, filename= 'final_merge.csv'):
    df1 = pd.read_csv(basepath/"2021-01-01_2021-04-01.csv")
    df2 = pd.read_csv(basepath/"2021-04-01_2021-08-01.csv")
    df3 = pd.read_csv(basepath/"2021-08-01_2022-01-01.csv")
    df4 = pd.read_csv(basepath/"2022-01-01_2022-04-01.csv")
    df5 = pd.read_csv(basepath/"2022-04-01_2022-08-01.csv")
    df6 = pd.read_csv(basepath/"2022-08-01_2023-04-01.csv")
    df =pd.concat([df1,df2,df3,df4,df5,df6])
    basepath = basepath.joinpath("Data")
    if save:
        df.to_csv(basepath/filename,index=False)
    return df

#################################################################
# filter records based on conditions 

def filter_records(df,num_per_vin,min_floor_price,min_miles=None,max_miles=None,save = True,filename='final_filter.csv'):
    #the goal of this is to select the specific rows of interest
    # I would setup a function to preprocess. 
    df.reset_index(inplace=True,drop=True)
    df.set_index('auction_id', inplace=True)
    #Drop all private auctions. 
    #Private auctions might have vastly different dynamics and as a result
    df=df[~df['auction_is_private']]

    #Small number of 2965
    df=df[~df['auction_is_bid_sale']]
    #set min floor price as a number at 100
    #df=df[df['auction_floor_price']>min_floor_price]
    required_vins = df["vehicle_vin"].value_counts().loc[lambda x: x>num_per_vin].index.values
    df = df[df["vehicle_vin"].isin(required_vins)]
    if min_miles!=None:
        df=df[df['auction_odometer']>=min_miles]
    if max_miles!=None:
        df=df[df['auction_odometer']<=max_miles]
    if min_floor_price!=None:
        df=df[df['auction_floor_price']>min_floor_price]

    #df=df[df['auction_is_green_light']==True]
    if save:
        df.to_csv(filename,index=False)

    return df

#####################################################################
# feature engineering function 

def create_Features(df):
    print("............")
    df['auction_start_time'] =  pd.to_datetime(df["auction_start_time"])
    df['auction_end_time'] =  pd.to_datetime(df["auction_end_time"])
    df['auction_time']=df['auction_end_time']-df['auction_start_time']
    df["auction_end_month"] =df['auction_end_time'].dt.month_name()
    df["auction_dow"] = df['auction_end_time'].dt.dayofweek
    df["auction_hour"] =df['auction_end_time'].dt.hour
    df["auction_year"]=df['auction_end_time'].dt.year
    df["auction_week"] = df['auction_end_time'].dt.isocalendar().week
    df['week_num']=df['auction_week']
    df.loc[df["auction_year"]==2022,'week_num']=df.loc[df["auction_year"]==2022,'week_num']+53
    df["auction_hour_of_week"]=df['auction_end_time'].dt.dayofweek * 24 + (df['auction_end_time'].dt.hour + 1)
    df["day_hour"] = "Day"+df["auction_dow"].astype(str)+"_hour"+df["auction_hour"].astype(str)
    df['auction_is_gross_sold'] = df['auction_is_gross_sold'].astype(int)
    df['auctionnum']=1
    df['auction_gross_run_number_calc']=df.groupby(['lister_user_id','auction_vehicle_id'])['auctionnum'].cumsum()
    df['auction_gross_run_number_calc_max']=df.groupby(['lister_user_id','auction_vehicle_id'])['auction_gross_run_number_calc'].transform('max')
    df['first_price']=df.loc[df['auction_gross_run_number_calc']==1,'auction_floor_price']
    df['first_price']=df.groupby(['lister_user_id','auction_vehicle_id'])['first_price'].transform('max')
    #df['auction_gross_run_number_max']=df.groupby(['auction_vehicle_id'])['auction_gross_run_number'].transform('max')
    #df['auction_gross_run_number_min']=df.groupby(['auction_vehicle_id'])['auction_gross_run_number'].transform('min')
    df['auction_floor_price_max']=df.groupby(['lister_user_id','auction_vehicle_id'])['auction_floor_price'].transform('max')
    df['auction_floor_price_min']=df.groupby(['lister_user_id','auction_vehicle_id'])['auction_floor_price'].transform('min')
    #df['auction_floor_price_min']=df.loc[df['auction_gross_run_number']==df['auction_gross_run_number_calc_min']
    df['percentage']= df['auction_floor_price']/df['first_price']
    df['percentage_max']=df.groupby(['lister_user_id','auction_vehicle_id'])['percentage'].transform('max')
    df['percentage_min']=df.groupby(['lister_user_id','auction_vehicle_id'])['percentage'].transform('min')
    df['auction_is_gross_sold'] = df['auction_is_gross_sold'].astype(int)
    df['auction_is_green_light']  =  df.auction_is_green_light.map({True: 1, False:0}) 
    df['auction_is_yellow_light'] = df.auction_is_yellow_light.map({True: 1, False:0})
    df['auction_is_blue_light'] = df.auction_is_blue_light.map({True: 1,False:0})
    df["auction_red_light_changed_to_green_in_past"] =  df["auction_red_light_changed_to_green_in_past"].map({True: 1,False:0})
    df['auction_gross_run_number_calc'] = df['auction_gross_run_number_calc'].astype("category")
    df['week_num_cat'] = df['week_num'].astype("category")
    df["time_discount"]=0 #time discount is initially set to 0
    df.loc[df['percentage_max']<=1.0,"time_discount"]=1 #for any values of percent_max <=1, time discount = 1 or true. So any values >1 mean time discount = 0
    # time discount is when the reserve price is reduced in consecutive auctions
    # building a logistic regression to see what drives time discounting behavior
    df["failed"]=0 #failed is initially set to 0
    df.loc[(df["auction_high_bid"]>df["auction_floor_price"])&(df["auction_is_gross_sold"]==0),"failed"]=1  
    df["liquidation"]=0 # liquidation is initially set to 0
    df.loc[df['auction_floor_price']<=1000,"liquidation"]=1 # for floor prices less than 1000, sellers
    # want to get rid of their vehicles, and this is a liquidation strategy 
    df.sort_values(['auction_vehicle_id', 'auction_gross_run_number_calc'], inplace=True)
    df['LaggedFloorPrice'] = df.groupby('auction_vehicle_id')['auction_floor_price'].shift(-1)
    return df

#############################################################################################
# preprocessing function for traning and predictions
def preprocess(df,cat_controls,num_controls):
    all_var=cat_controls+num_controls
    df=df[all_var] 
    # one hot encoding
    
    df_dummy =pd.get_dummies(df[cat_controls], drop_first=True)
    df_final =  pd.concat([df_dummy,df[num_controls]],axis=1)
    #df_final = df_final.dropna()
    return df_final 

##########################################################################################
# Train_test_split and modelling

def modelling(df_final, training_weeks, output_variable, model_name):
    
    if output_variable == 'auction_sale_amount':
        df_final = df_final[df_final["auction_is_gross_sold"]==1]
        
    train = df_final[df_final["week_num"] <= training_weeks]
    test = df_final[df_final["week_num"] > training_weeks]
    print("Training data shape:", train.shape, "Testing data shape:", test.shape)
    print("Model:", model_name)

    train_X = train.drop(["week_num", "auction_floor_price", 'auction_sale_amount',"auction_bid_count","auction_is_gross_sold"], axis=1)
    test_X = test.drop(["week_num", "auction_floor_price", 'auction_sale_amount',"auction_bid_count","auction_is_gross_sold"], axis=1)

    if output_variable == "auction_floor_price":
        train_y = train["auction_floor_price"]
        test_y = test["auction_floor_price"]
    elif output_variable == 'auction_sale_amount':
        train_y = train['auction_sale_amount']
        test_y = test['auction_sale_amount']
    elif output_variable == "auction_bid_count":
        train_y = train['auction_bid_count']
        test_y = test['auction_bid_count']
        
    del df_final
    del train
    del test
    gc.collect()

    if model_name == "Random Forest":
        model = RandomForestRegressor()
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor()
    elif model_name == "XGBoost":
        model = xgb.XGBRegressor()
    elif model_name == "LightGBM":
        model = lgb.LGBMRegressor()
    elif model_name == "CatBoost":
        model = CatBoostRegressor(logging_level='Silent')
    elif model_name == "AdaBoost":
        model = AdaBoostRegressor()

    start_time = time.time()
    model.fit(train_X, train_y)
   

    # Predict on the train and test data
    train_predictions = model.predict(train_X)
    test_predictions = model.predict(test_X)
    
    # end time after predictions
    end_time = time.time()
    total_time = end_time - start_time

    # R2 of train and test
    r2_train = r2_score(train_y, train_predictions)
    r2_test = r2_score(test_y, test_predictions)
    
    # Calculate RMSE
    train_rmse = mean_squared_error(train_y, train_predictions, squared=False)
    test_rmse = mean_squared_error(test_y, test_predictions, squared=False)
    
    # Calculate MSE
    train_mse = mean_squared_error(train_y, train_predictions)
    test_mse = mean_squared_error(test_y, test_predictions)
    
  

    result = {
        'Model': model_name,
        'Time Taken': total_time,
        'Train R2': r2_train,
        'Test R2': r2_test,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train MSE': train_mse,
        'Test MSE': test_mse
    }
    
    return result,model

###########################################################################################

def get_predictions(df_final,df,output_variable,model,training_weeks):
    
    train = df_final[df_final["week_num"] <= training_weeks]
    test = df_final[df_final["week_num"] > training_weeks]
    train_X = train.drop(["week_num", "auction_floor_price", 'auction_sale_amount',"auction_bid_count"], axis=1)
    test_X = test.drop(["week_num", "auction_floor_price", 'auction_sale_amount',"auction_bid_count"], axis=1)
    
    del train
    del test
    gc.collect()
    
    loaded_model = pickle.load(open(model, 'rb'))
    train_predictions = loaded_model.predict(train_X)
    test_predictions = loaded_model.predict(test_X)
    
    # attaching it to original dataframe
    
    train_maindf = df[df["week_num"] <= training_weeks]
    test_maindf = df[df["week_num"] > training_weeks]
    
    var = output_variable+"_preds"
    train_maindf[var] = train_predictions
    test_maindf[var] = test_predictions
    
    df_final_maindf =  pd.concat([train_maindf,test_maindf])
    
    # Move the output variable column to the last position
    columns = df_final_maindf.columns.tolist() 
    columns.remove(output_variable)         
    columns.append(output_variable)
    
    df_final_maindf = df_final_maindf[columns]
    
    return df_final_maindf

