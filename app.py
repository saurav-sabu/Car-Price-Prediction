import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


df = pickle.load(open("df.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


nav = st.sidebar.radio(
    "Navigation", ["About The Project", "Make Prediction", "Contribute To Dataset "])

if nav == "About The Project":
    st.markdown("# About The Company")
    st.image("CarDekho-logo.jpg")
    st.markdown("""
    About CarDekho
CarDekho.com is India's leading car search venture that helps users buy cars that are right for them. Its website and app carry rich automotive content such as expert reviews, detailed specs and prices, comparisons as well as videos and pictures of all car brands and models available in India. The company has tie-ups with many auto manufacturers, more than 4000 car dealers and numerous financial institutions to facilitate the purchase of vehicles.

CarDekho.com has launched many innovative features to ensure that users get an immersive experience of the car model before visiting a dealer showroom. These include a Feel The Car tool that gives 360-degree interior/exterior views with sounds of the car and explanations of features with videos; search and comparison by make, model, price, features; and live offers and promotions in all cities. The platform also has used car classifieds wherein users can upload their cars for sale, and find used cars for buying from individuals and used car dealers.

Besides the above consumer product features, CarDekho.com provides a rich array of tech-enabled tools to OE manufacturers and car dealers. These include apps for dealer sales executives to manage leads, cloud services for tracking sales performance, call tracker solution, digital marketing support, virtual online showroom and outsourced lead management operational process for taking consumers from enquiry to sale.

Our vision is to construct a complete ecosystem for consumers and car manufacturers, dealers and related businesses such that consumers have easy and complete access to not only buying and selling cars, but also manage their entire ownership experience, be it accessories, tyres, batteries, insurance or roadside assistance.

The company has expanded to Southeast Asia with the launch of Zigwheels.ph, Zigwheels.my and Oto.com. It also has a presence in the UAE with Zigwheels.ae

To diversify our product offerings we have ventured into car insurance business through InsuranceDekho.Com
<hr> 
    """,unsafe_allow_html=True)

    st.markdown("# About The Project")
    st.markdown(
        "This dataset contains information about used cars.This data is collected from CarDekho website using webscrapping.")
    st.markdown("""The columns in the given dataset are as follows:

        name
        year
        selling_price
        km_driven
        fuel
        seller_type
        transmission
        Owner
        """, unsafe_allow_html=True)

    st.markdown("""<b>name:-</b> Name of the cars""", unsafe_allow_html=True)
    st.markdown("""year:- Year of the car when it was bought""")
    st.markdown("""selling_price:- Price at which the car is being sold""")
    st.markdown("""km_driven:- Number of Kilometres the car is driven""")
    st.markdown(
        """fuel:- Fuel type of car (petrol / diesel / CNG / LPG / electric)""")
    st.markdown("""seller_type:- Tells if a Seller is Individual or a Dealer""")
    st.markdown(
        """transmission:- Gear transmission of the car (Automatic/Manual)""")
    st.markdown("""owner:- Number of previous owners of the car.""")

    st.markdown(
        """### We need to Predict the selling price of the car using our model""")

elif nav == "Make Prediction":
    st.title("Car Price Prediction")

    year = st.number_input("Year", 1970, 2022)
    year = 2022 - year
    price = st.number_input("What is the Showroom Price(in lakhs)", 1, 100)
    km_driven = st.number_input("How many Kilometers Drived", 0)
    owner = st.number_input(
        "How many owners previously owned the car(0, 1 or 3)", 0, 3)
    fuel_type = st.selectbox("Fuel type", df["Fuel_Type"].unique())
    seller_type = st.selectbox("Seller type", df["Seller_Type"].unique())
    transmission_type = st.selectbox(
        "Transmission type", df["Transmission"].unique())

    if st.button("Predict"):
        if fuel_type == "Petrol":
            Fuel_Type_Petrol = 1
            Fuel_Type_Diesel = 0
        elif fuel_type == "Diesel":
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 1
        else:
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 0

        if seller_type == "Individual":
            Seller_Type_Individual = 1
        else:
            Seller_Type_Individual = 0

        if transmission_type == "Manual":
            Transmission_Manual = 1
        else:
            Transmission_Manual = 0

        query = np.array([price, km_driven, owner, year, Fuel_Type_Petrol,
                          Fuel_Type_Diesel, Seller_Type_Individual, Transmission_Manual])
        query = query.reshape(1, 8)
        query = model.predict(query)
        st.success(f"Price of the Car: {np.round((query),2)[0]}",)

else:
    st.title("Contribute to Dataset ")

    
    upload = st.checkbox("Want to upload csv file")    
    
    try:
        if upload:
            df1 = st.file_uploader("Upload csv files")
            df2 = pd.read_csv(df1)
            df = pd.concat([df,df2])
            st.write(df.shape[0])
    except Exception as e:
        st.error("Please Upload the csv file")

    show = st.checkbox("Show Top Records")

    if show:
        num = st.slider(
            f"Enter the number of record you want to see", 1, df.shape[0])
        st.table(df.head(num))
    
    show1 = st.checkbox("Show Bottom Records")

    if show1:
        num1 = st.slider(
            f"Enter the number of record you want to see", 1, df.shape[0],key="s")
        st.table(df.tail(num1))

 



   
