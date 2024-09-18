import pickle
import pandas as pd
import numpy as np
import streamlit as st
from PIL.ImageOps import expand
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder
import requests


with open('xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('std_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def extract_city_code(df):
    df["City_Code"] = df["Delivery_Person_Id"].str.split("RES", expand=True)[0]

def extract_date_features(df):
    df['day'] = df['Order_Date'].dt.day
    df['month'] = df['Order_Date'].dt.month
    df['quarter'] = df['Order_Date'].dt.quarter
    df['year'] = df['Order_Date'].dt.year
    df['day_of_week'] = df['Order_Date'].dt.day_of_week.astype(int)
    df['is_month_start'] = df['Order_Date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['Order_Date'].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df['Order_Date'].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df['Order_Date'].dt.is_quarter_end.astype(int)
    df['is_year_start'] = df['Order_Date'].dt.is_year_start.astype(int)
    df['is_year_end'] = df['Order_Date'].dt.is_year_end.astype(int)
    df['is_weekend'] = np.where(df['day_of_week'].isin([5,6]), 1, 0)

def calculate_time_diff(df):
    df['Time_Orderd'] = pd.to_timedelta(df['Time_Orderd'])
    df['Time_Order_Picked'] = pd.to_timedelta(df['Time_Order_Picked'])

    df['Time_Order_Picked_Formatted'] = df['Order_Date'] + \
                                       np.where(df['Time_Order_Picked'] < df['Time_Orderd'], pd.DateOffset(days=1), pd.DateOffset(days=0)) + \
        df['Time_Order_Picked']

    df['Time_Ordered_Formatted'] = df['Order_Date'] + df['Time_Orderd']

    df['Time_Order_Picked_Formatted']=pd.to_datetime(df['Time_Order_Picked_Formatted'])

    df['Order_Prepare_Time'] = (df['Time_Order_Picked_Formatted'] - df['Time_Ordered_Formatted']).dt.total_seconds() / 60

    # Handle null values by filling with the median
    df['Order_Prepare_Time'].fillna(df['Order_Prepare_Time'].median(), inplace=True)

    # Drop all the time & date related columns
    df.drop(['Time_Orderd', 'Time_Order_Picked', 'Time_Ordered_Formatted', 'Time_Order_Picked_Formatted', 'Order_Date'], axis=1, inplace=True)



def calculate_distance(df):
    df['Distance']=np.zeros(len(df))
    restaurant_coordinates=df[['Restaurant_Latitude','Restaurant_Longitude']].to_numpy()
    delivery_location_coordinates=df[['Delivery_Location_Latitude','Delivery_Location_Longitude']].to_numpy()
    df['Distance'] = np.array([geodesic(restaurant, delivery) for restaurant, delivery in zip(restaurant_coordinates, delivery_location_coordinates)])
    df['Distance']= df['Distance'].astype("str").str.extract('(\d+)').astype("int64")


def label_encoding(df):
    categorical_columns = df.select_dtypes(include='object').columns
    label_encoder = LabelEncoder()
    df[categorical_columns] = df[categorical_columns].apply(lambda col: label_encoder.fit_transform(col))


# Starting to build a Streamlit app
st.title('Food Delivery Time Prediction')

# User input for each feature
delivery_person_id = st.text_input('Delivery Person ID', 'BANGRES19DEL01')
age = st.number_input('Delivery Person Age', min_value=18, max_value=65, value=30)
ratings = st.number_input('Delivery Person Ratings', min_value=1.0, max_value=5.0, value=4.5)
order_date = st.date_input('Order Date')
time_ordered = st.time_input('Time Ordered')
time_order_picked = st.time_input('Time Order Picked')
weather = st.selectbox('Weather Conditions', ['Sunny','Cloudy','Rainy','Foggy'])
traffic = st.selectbox("Road Traffic Density",['Low','Medium','High'])
vehicle_condition = st.number_input('Vehicle Condition', min_value=0, max_value=10, value=7)
order_type = st.selectbox('Type of Order', ['Snack', 'Meal', 'Drinks', 'Buffet'])
vehicle_type = st.selectbox('Type of Vehicle', ['Bike', 'Scooter', 'Car', 'Truck'])
multiple_deliveries = st.number_input('Multiple Deliveries', min_value=0, max_value=5, value=0)
festival = st.selectbox('Festival', ['No', 'Yes'])
city = st.selectbox('City', ['Urban', 'Semi-Urban', 'Metropolitan'])

restaurant_address = st.text_input('Restaurant Address')
delivery_address = st.text_input('Delivery Address')


def get_lat_long_opencage(address, api_key):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={address}&key={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        if data['results']:
            latitude = data['results'][0]['geometry']['lat']
            longitude = data['results'][0]['geometry']['lng']
            return latitude, longitude
        else:
            return None, None
    else:
        return None, None

api_key = 'ff2a180dcfed458390ed03764ce7a0e9'

restaurant_lat,restaurant_long = get_lat_long_opencage(restaurant_address, api_key)
delivery_lat,delivery_long = get_lat_long_opencage(delivery_address, api_key)

def set_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url({image_url});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function to set the background image
image_url = 'https://www.google.com/url?sa=i&url=https%3A%2F%2Fstock.adobe.com%2Fsearch%3Fk%3Dfood%2Bdelivery&psig=AOvVaw0xm517YZ9lHOlIbkAhvqPv&ust=1726533935862000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCMjVpKGexogDFQAAAAAdAAAAABAE'
set_background_image(image_url)


if st.button("Get ETA for Delivery!"):
    # Prepare input data for the model
    input_data = pd.DataFrame({
        'Delivery_Person_Id': [delivery_person_id],
        'Delivery_Person_Age': [age],
        'Delivery_Person_Ratings': [ratings],
        'Restaurant_Latitude': [restaurant_lat],
        'Restaurant_Longitude': [restaurant_long],
        'Delivery_Location_Latitude': [delivery_lat],
        'Delivery_Location_Longitude': [delivery_long],
        'Order_Date': [order_date],
        'Time_Orderd': [time_ordered],
        'Time_Order_Picked': [time_order_picked],
        'Weather_Conditions': [weather],
        'Road_Traffic_Density': [traffic],
        'Vehicle_Condition': [vehicle_condition],
        'Type_Of_Order': [order_type],
        'Type_Of_Vehicle': [vehicle_type],
        'Multiple_Deliveries': [multiple_deliveries],
        'Festival': [festival],
        'City': [city]
    })


    input_data['Order_Date'] = pd.to_datetime(input_data['Order_Date'])
    input_data['Order_Year'] = input_data['Order_Date'].dt.year
    input_data['Order_Month'] = input_data['Order_Date'].dt.month
    input_data['Order_Day'] = input_data['Order_Date'].dt.day
    input_data['Time_Orderd'] = pd.to_datetime(input_data['Time_Orderd'], format='%H:%M:%S').dt.hour
    input_data['Time_Order_Picked'] = pd.to_datetime(input_data['Time_Order_Picked'], format='%H:%M:%S').dt.hour



    extract_city_code(input_data)
    extract_date_features(input_data)
    calculate_time_diff(input_data)
    calculate_distance(input_data)
    label_encoding(input_data)

    input_data=input_data.drop(['Order_Year', 'Order_Month', 'Order_Day','Delivery_Person_Id'],axis=1)

    # Scale the input data using the loaded scaler
    scaled_data = scaler.transform(input_data)

    # Make prediction using the loaded model
    prediction = model.predict(scaled_data)

    # Display the prediction
    st.write(f'Your Food will arrive in (minutes) : {round(prediction[0])}')