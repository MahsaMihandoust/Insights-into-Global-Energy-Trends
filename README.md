# Insights-into-Global-Energy-Trends
## Problem Statement: 
The primary challenge addressed in this project is the accurate prediction and analysis of sustainable energy indicators to support data-driven decision-making and policy formulation.

## Key objectives include: 
1.	Classifying Countries with a potential for building the energy infrastructure on renewable energy. 
2.	Predicting future electricity consumption patterns to aid in planning and resource allocation.
3.	Forecasting carbon emissions to support climate change mitigation strategies.

## Methodology:
1.	Feed Forward Neural Network: FNN is used to analyze past electricity consumption sequences and predict future values, capturing temporal dependencies and trends.
2.	ARIMA Model: ARIMA models account for overall trends in the data, providing forecasts for various energy indicators.
3.	Smoothing Methods: Techniques such as moving averages and exponential smoothing applied to identify long-term trends and reduce noise in the dataset, facilitating clearer insights into energy consumption patterns

# Key Features of the Dataset
The "Global Data on Sustainable Energy (2000-2020)" dataset comprises a comprehensive set of variables critical for the analysis and prediction of sustainable energy trends across various countries. The dataset contains 3650 entries and 21 features, encapsulating a broad range of indicators over two decades. The key features of this dataset are listed below:
1.	Entity
2.	Year
3.	Access to Electricity (% of Population)
4.	Access to Clean Fuels for Cooking (% of Population)
5.	Renewable Electricity Generating Capacity Per Capita
6.	Financial Flows to Developing Countries (US $)
7.	Renewable Energy Share in Total Final Energy Consumption (%)
8.	Electricity from Fossil Fuels (TWh)
9.	Electricity from Nuclear (TWh)
10.	Electricity from Renewables (TWh)
11.	Low-Carbon Electricity (% of Total Electricity)
12.	Primary Energy Consumption Per Capita (kWh/Person)
13.	Energy Intensity Level of Primary Energy (MJ/$2011 PPP GDP)
14.	Carbon Dioxide Emissions Per Capita (Metric Tons)
15.	Renewables (% of Equivalent Primary Energy)
16.	GDP Growth (Annual %)
17.	GDP Per Capita
18.	Population Density (Persons/Km²)
19.	Land Area (Km²)
20.	Latitude
21.	Longitude
    
# Data Cleaning and Preprocessing
1. Data Filtering

•	Objective: Refine the dataset to include only those records relevant to the specified countries for renewable energy analysis.
•	Process: The dataset is filtered to focus solely on the countries of interest: Australia, Canada, China, Denmark, Germany, Poland, United Arab Emirates, United Kingdom, United States, and Turkey. This ensures that subsequent analysis is pertinent to these specific regions.
•	Result: The dataset is reduced to 210 entries specific to the selected countries.

2. Inspection of Data
•	Objective: Assess the overall structure and completeness of the dataset.
•	Process: An initial inspection of the dataset provides an overview of its columns and the count of non-null entries in each. This step is crucial for understanding the data’s quality and identifying any immediate issues.

3. Missing Values Analysis
•	Objective: Identify and visualize columns with missing values.
•	Process: A visual displays the count of missing values across each column. This visualization helps in pinpointing which columns are incomplete and may require attention or imputation using mean values.

4. Data Classification
•	Objective: Convert continuous numerical data into categorical classes based on predefined criteria.
•	Process: A threshold is established to classify the renewable energy share into binary categories. This transformation turns the continuous variable into a categorical one, facilitating easier analysis and interpretation of renewable energy levels relative to the threshold.
•	Result: A new column, ‘Renewable_energy_classified’, is created to categorize renewable energy share as either 1 (above threshold) or 0 (below threshold).

5. Data Review
•	Objective: Confirm the successful application of preprocessing steps.
•	Process: The updated dataset is reviewed to ensure that the new classification column has been correctly added and that the data reflects the intended preprocessing transformations.

# Exploratory Data Analysis
### Breakdown of the Carbon Emissions Bar Chart:

![image](https://github.com/user-attachments/assets/6b9e7c22-5b2f-44e1-a066-6c172699b55c)

### Trend of CO2 Emissions Over Time:

![image](https://github.com/user-attachments/assets/f705bef8-669b-4723-a91d-25914aaa9393)

### Changes in Renewable Energy Share Over the Years:

![image](https://github.com/user-attachments/assets/b75855cc-c3ba-4fd6-99d6-62eed74ddee2)




# Feature Selection for Neural Network Model
To identify the most pertinent features, numeric variables associated with energy consumption and renewable energy metrics were selected. These variables included measures such as energy consumption per capita, access to clean fuels, and carbon emissions, among others.

A ‘RandomForestRegressor’ was employed to evaluate the importance of each feature in predicting the target variable. This model facilitated the assessment of feature significance, resulting in the ranking of features according to their influence on the target variable

![image](https://github.com/user-attachments/assets/01edc249-e225-413f-b120-376bac53bf5e)

# Using Neural Network Model
## Model Training and Configuration

The neural network model, implemented using a Multi-Layer Perceptron (MLP) classifier, aimed to classify countries based on their potential for developing energy infrastructure centered on renewable energy. Various hidden layer configurations were tested, ranging from a single neuron to up to 1000 neurons, to identify the optimal model setup. The MLP classifier was trained with these configurations, and its performance was evaluated based on both training and validation accuracy.

## Evaluation of Model Performance
The model's effectiveness in classifying countries' potential for renewable energy infrastructure was assessed across different configurations:

•	Hidden Layer Configuration (1 neuron): Training accuracy was 98.33%, and validation accuracy was 88.75%.

•	Hidden Layer Configuration (2 neurons): Training accuracy was 98.33%, and validation accuracy was 91.25%.

•	Hidden Layer Configuration (5 neurons): Training accuracy was 98.33%, and validation accuracy was 91.25%.

•	Hidden Layer Configuration (10 neurons): Training accuracy was 100%, and validation accuracy was 96.25%.

•	Hidden Layer Configuration (20 neurons): Training accuracy was 100%, and validation accuracy was 96.25%.

•	Hidden Layer Configuration (50 neurons): Training accuracy was 100%, and validation accuracy was 93.75%.

•	Hidden Layer Configuration (500 neurons): Training accuracy was 100%, and validation accuracy was 96.25%.

•	Hidden Layer Configuration (1000 neurons): Training accuracy was 100%, and validation accuracy was 95.00%.

The configuration with 10 hidden neurons was found to be optimal, achieving the highest validation accuracy of 96.25%. This high accuracy indicates that the model effectively distinguishes between countries with varying potentials for renewable energy infrastructure.

# Energy Consumption Forecasting Using ARIMA, Exponential Smoothing, and Moving Average Models

•	ARIMA models were fitted using the (1, 1, 1) order for each country. The fitted models were then used to forecast energy consumption for the next five years (2021-2025).

•	The results for China, for instance, show a steady increase in energy consumption, with the forecasted values ranging from 29,846 kWh/person in 2021 to 32,478 kWh/person in 2025.

![image](https://github.com/user-attachments/assets/ff3f6d30-6221-4d63-bbf3-ddb1ee267079)

![image](https://github.com/user-attachments/assets/605c30a1-1aa7-4ed8-a111-a00dc0f27db0)


# C02 Emission Energy Consumption Forecasting Using ARIMA, Exponential Smoothing, and Moving Average Models

An ARIMA model is fitted to the historical CO2 emission data for each country. This model captures the underlying patterns and trends in the data. 
The ARIMA model used in this analysis has an order of (1, 1, 1), which implies one autoregressive term, one differencing term, and one moving average term. These parameters are chosen to balance model complexity and predictive power. 

![image](https://github.com/user-attachments/assets/32bc4004-ac71-4da5-b9d7-4eb82eb9edec)

![image](https://github.com/user-attachments/assets/b1930d13-8e23-46a5-b5a3-6de321b74880)


# Conclusion
•	As shown by the neural network,UAE and the United states are not able to contribute to renewable energy due to the lack of infrastructure.

•	ARIMA and Exponential Smoothing models indicate that energy consumption in China is expected to increase to over 30,000 kW per person by 2025.

•	ARIMA forecasts an increase in CO2 emissions of up to 1.2 kt by 2025, while Exponential Smoothing and Moving Average models project stable emissions at 1 kt.





