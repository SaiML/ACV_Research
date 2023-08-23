#The goal of this package is to be systematic in ensuring that we have common processes. 

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer, LineLocation
import utils


#Using Sai's version
def filter_records(df,num_per_vin,min_floor_price,min_miles=None,max_miles=None,save = True,filename='final_filter.csv', drop=False):
    #the goal of this is to select the specific rows of interest
    # I would setup a function to preprocess. 
  
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
    if drop:
        print("dropping rare missing values")
        df, summary=utils.missing_values_summary(df,drop=True)
    #df=df[df['auction_is_green_light']==True]
    if save:
        df.to_csv(filename,index=False)
    df.reset_index(inplace=True,drop=True)

    return df


#####################################################################
# feature engineering function 

def create_features(df):
    print("............")
    df['auction_start_time'] =  pd.to_datetime(df["auction_start_time"])
    df['auction_end_time'] =  pd.to_datetime(df["auction_end_time"])
    df['auction_time']=df['auction_end_time']-df['auction_start_time']
    df["auction_end_month"] =df['auction_end_time'].dt.month_name()
    df["auction_dow"] = df['auction_end_time'].dt.dayofweek
    df["auction_hour"] =df['auction_end_time'].dt.hour
    df["auction_year"]=df['auction_end_time'].dt.year
    df["day_of_month"] = df['auction_end_time'].dt.day
    df["auction_week"] = df['auction_end_time'].dt.isocalendar().week
    df['week_num']=df['auction_week']
    df['is_holiday']=df['auction_end_time'].apply(utils.is_holiday)
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
    df['auction_premium']=df['auction_ending_amount']-df['auction_floor_price']
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
    df['lagged_auction_floor_price'] = df.groupby('auction_vehicle_id')['auction_floor_price'].shift(-1)
    df['previous_floor_price'] = df.groupby('auction_vehicle_id')['auction_floor_price'].shift(+1)
    df['lagged_auction_end_time'] = df.groupby('auction_vehicle_id')['auction_end_time'].shift(-1)
    df['previous_auction_end_time'] = df.groupby('auction_vehicle_id')['auction_end_time'].shift(+1)
    df['between_auctions']= df['auction_end_time'] -df['previous_auction_end_time'] 
    df['between_auctions']=df['between_auctions'].dt.total_seconds()/ 3600
    df.reset_index(inplace=True,drop=True)
    df['days_since_start'] = (df['auction_end_time'] - df['auction_end_time'].min()).dt.days
    df['hours_since_start'] = (df['auction_end_time'] - df['auction_end_time'].min()).dt.total_seconds() / 3600
    df['hours_since_start']=df['hours_since_start'].astype(int)
    return df


def generate_stacked_reg(title, exps, dropcol=[], cov_order=[], order=False, siglevels=[0.05, 0.01, 0.001]):
    """
    Perform stacked regression analysis on a list of experiments.

    Parameters:
        title (str): Title of the analysis.
        exps (list): List of experiment dictionaries, each containing details about the experiment.
        dropcol (list, optional): Columns to be dropped from the dummy variable DataFrame. Default is an empty list.
        cov_order (list, optional): The desired order of covariates in the summary table. Default is an empty list.
        order (bool, optional): If True, the covariate order in the summary table will be set based on `cov_order`.
                                Default is False.
        siglevels (list, optional): Significance levels for displaying stars in the summary table. Default is [0.05, 0.01, 0.001].

    Returns:
        tuple: A tuple containing the regression results, the Stargazer summary table, and the predictor variable columns.

    Example:
        See example at the end of the code.

    """
    results = []
    pseudo_r_list = []
    column_names = []

    for e in exps:
        # Check if dropcol list is empty or not and set the 'drop_first' flag accordingly.
        if len(dropcol) > 0:
            drop_first = False
        else:
            drop_first = True

        print("Running ", e['name'])
        # Create dummy variables for categorical columns in the experiment's DataFrame.
        df_cat = pd.get_dummies(e['df'][e['cat']], columns=e['cat'], drop_first=drop_first)

        # Drop specified columns from the dummy variable DataFrame.
        for col in dropcol:
            if col in df_cat.columns:
                df_cat.drop(col, axis=1, inplace=True)

        # Concatenate dummy variable DataFrame and numerical columns to form the predictor variables (X).
        X = pd.concat([df_cat, e['df'][e['num']]], axis=1)
        Xs = sm.add_constant(X)  # Append a constant column (required for regression in statsmodels).

        # Add the experiment name to the column_names list.
        column_names.append(e['name'])

        # Check for missing values in the predictor variables.
        assert X.isna().sum().sum() == 0

        # Perform regression based on the experiment type.
        if e['type'] == 'bin':
            exp = sm.Logit(e['df'][e['dv']], Xs).fit()  # Logistic regression
            pseudo_r = exp.prsquared
            pseudo_r_list.append(round(pseudo_r, 4))
        elif e['type'] == 'ols':
            exp = sm.OLS(e['df'][e['dv']], Xs).fit()  # Ordinary Least Squares (OLS) regression

        results.append(exp)

        # Add the experiment name to the column_names list again to account for pseudo R-squared values.
        column_names.append(e['name'])

    # Create a Stargazer object to generate a summary table.
    stargazer = Stargazer(results)
    # stargazer.custom_columns(column_names, [1 for x in range(len(column_names))])
    stargazer.significance_levels(siglevels)

    # Set the covariate order if the 'order' flag is True.
    if order:
        stargazer.covariate_order(cov_order)

    # Get the names of the predictor variable columns.
    predictor_columns = X.columns

    return results, stargazer, predictor_columns

