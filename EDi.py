import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st 
st.set_page_config(
    page_title = "Data Analysis App",
    layout = "wide"
) 
st.title("Tracking Climate Change: Analyzing CO₂ Emissions and Global Temperature Trends")
st.write("""
Welcome to Tracking Climate Change App.
It is a data-driven exploration tool designed for climate analysts, educators, and policy shapers.
It reveals how rising CO₂ emissions are intertwined with global temperature trends—turning historical climate data into clear, actionable insights.
Built on two key datasets—atmospheric CO₂ concentrations and temperature anomalies across countries—this app helps users:
- **Visualize** long-term climate shifts from 1958 to 2024
- **Detect** unusual spikes or dips tied to global events (Covid-19)
- **Compare warming** patterns by region and spotlight the most at-risk countries
- **Predict** future temperature trajectories based on changes in CO₂ emissions
- **Simulate** "what-if" policy outcomes, like reducing emissions by 20%
- **Craft data-backed** recommendations for climate action
- **Using** sidebars for navigation and gathering user inputs
- Displaying **text**, markdown and data tables
- **Generating** fata with numpy and pandas
Whether you’re curious about the past, concerned about the future, or working on solutions right now—this app turns climate data into stories the world can act on.
Each section of this code is commented to help you understand
""")
st.sidebar.header("User Input Feature")
name = st.sidebar.text_input("Enter your name:", "Streamlit User")
st.header(f"Hello {name}! Lets create something interractive")
# Load CO2 data
import pandas as pd
df = pd.read_csv('carbon_emmission.csv')
df = df.rename(columns={"Value": "CO2_ppm"})
st.subheader("Carbon Emission Data")
st.write(df.head())

df.head()
#Filter valid CO2 readings (values > 200 ppm)
CO2_df = df[df['CO2_ppm'] > 200].copy()
# Convert to datetime
CO2_df['Date'] = pd.to_datetime(CO2_df['Date'].str.replace('M', '-'))
st.subheader("Filtered CO2 Data")
st.write(CO2_df.head())
# Extract year/month
CO2_df['Year'] = CO2_df['Date'].dt.year
CO2_df['Month'] = CO2_df['Date'].dt.month
st.subheader("CO2 Data with Year and Month")
st.write(CO2_df.head())
# Annual averages
annual_CO2 = CO2_df.groupby('Year')['CO2_ppm'].mean().reset_index().round(3)
st.subheader("Annual Average CO2 Data Table")
st.write(annual_CO2.head())
# Sidebar for user input
st.sidebar.header("Data Visualization Annual Average CO2 Data")
st.subheader("Data Visualization for Average CO2 Data ")
Annual_CO2_Data = st.sidebar.slider("Number of Annual CO2 Data:", min_value=5, max_value=len(annual_CO2), value=10, step=5)
st.write(f"Displaying {Annual_CO2_Data} years of annual average CO2 data.")
annual_CO2 = annual_CO2.head(Annual_CO2_Data)

# Plotting
import matplotlib.pyplot as plt
# Plot trend
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(annual_CO2['Year'], annual_CO2['CO2_ppm'], 'r-o')
ax.set_title('Global CO₂ Concentration (1958-2023)', fontsize=14)
ax.set_ylabel('CO₂ (parts per million)')
ax.set_xlabel('Year')
ax.grid(alpha=0.3)
st.pyplot(fig) 
st.write("The graph shows a clear upward trend in CO₂ levels from 1958 to 2023, with a slight dip around 2020 likely due to reduced emissions during the COVID-19 pandemic.")
st.sidebar.info("This section visualizes the annual average CO2 concentrations from 1958 to 2023. Use the slider to adjust how many years of data to display.")

#load temperature data
st.subheader("Global Temperature Data")
dy_temp = pd.read_csv('temperature.csv')
st.write(dy_temp.head())
# Rename only columns that start with 'F' by removing the 'F' prefix
# Get the original column names before renaming
original_columns = dy_temp.columns
dy_temp.rename(
    columns={col: int(col[1:]) for col in original_columns if isinstance(col, str) and col.startswith('F')}, inplace=True
)
# Print the first 5 rows of the cleaned DataFrame
st.write(dy_temp.head())
# Replace all NaN values with 0 in the entire DataFrame
dy_temp = dy_temp.fillna(0)
st.subheader("Cleaned Global Temperature Data Extracting Year Columns")
# Extract just the year columns (after renaming)
year_columns = [col for col in dy_temp.columns if isinstance(col, int)]

# Melt the DataFrame from wide to long format
dy_long = dy_temp.melt(
    id_vars='Country',                 # Keep 'Country' as is
    value_vars=year_columns,          # All the year columns
    var_name='Year',                  # New column name for year
    value_name='Temperature_change'   # New column name for the temperature anomaly
)
#  Show the result
st.write(dy_long.head())

# Show average temperature change per year
global_temp_change = dy_long.groupby('Year')['Temperature_change'].mean().reset_index().round(3)
st.write(global_temp_change.head())

st.sidebar.header("Average Global Temperature Change Data")
st.subheader("Average Global Temperature Change Data Table")
Annual_Temp_Data = st.sidebar.slider("Number of Annual Temp Data:", min_value=5, max_value=len(global_temp_change), value=10, step=5)
st.write(f"Displaying {Annual_Temp_Data} years of annual average temperature change data.")
global_temp_change = global_temp_change.head(Annual_Temp_Data)

#Plot the global average temperature change per year (aggregated across all countries) from 1960–2022.
# Filter for 1960-2022
global_temp_change = global_temp_change[(global_temp_change['Year'] >= 1960) & (global_temp_change['Year'] <= 2022)]
plt.figure(figsize=(10,8))
plt.plot(global_temp_change['Year'], global_temp_change['Temperature_change'], 'r-o')
plt.title('Global Average Temperature Change (1960-2022)', fontsize=14)
plt.ylabel('Temperature Change (°C)')
plt.xlabel('Year')
plt.grid(alpha=0.3)
st.pyplot(plt)
st.write("The graph illustrates a general upward trend in global average temperature change from 1960 to 2022, indicating a warming climate. Notable fluctuations may correspond to significant climatic events or anomalies.")
st.sidebar.info("This section visualizes the annual average global temperature change from 1960 to 2022. Use the slider to adjust how many years of data to display.")
st.subheader("Graph between CO2 Levels and Global Temperature Change")
# Merge CO2 and temperature data on Year
merged_data = pd.merge(annual_CO2, global_temp_change, on='Year', how='inner')
st.write(merged_data.head())
st.sidebar.header("Merged CO2 and Temperature Data")
st.subheader("Merged CO2 and Temperature Data Table")
Merge_Av_Data = st.sidebar.slider("Number of Merge CO2 and Temp Data:", min_value=5, max_value=len(merged_data), value=5, step=1)
st.write(f"Displaying {Merge_Av_Data} graph of CO2 and temp Av. data.")
merged_data = merged_data.head(Merge_Av_Data)
#creating a plot
# Create figure and axes
fig, ax1 = plt.subplots(figsize=(12, 6))
# Plot CO₂ data (left axis)
color = 'tab:red'
ax1.set_xlabel('Year')
ax1.set_ylabel('CO₂ Concentration (ppm)', color=color)
ax1.plot(merged_data['Year'], merged_data['CO2_ppm'], color=color, marker='o', linestyle='-')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(alpha=0.3)

# Create second y-axis for temperature
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Temperature Anomaly (°C)', color=color)
ax2.plot(merged_data['Year'], merged_data['Temperature_change'], color=color, marker='s', linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)
plt.legend(labels=['global_temperature_change'], title="CO2 vs Temp anomaly", loc='upper left')

# Add title and legend
plt.title('CO₂ Concentration vs Global Temperature Anomaly (1960-2022)')
fig.tight_layout()
plt.grid(alpha=0.3)
st.pyplot(fig)

st.subheader("Top 10 Countries with Highest Temperature Anomalies from 2003-2022")
# Filter for 2003-2022
recent_temp = dy_long[(dy_long['Year'] >= 2003) & (dy_long['Year'] <= 2022)]
contry_avg_temp = recent_temp.groupby('Country')['Temperature_change'].mean().reset_index()
top10_countries = contry_avg_temp.sort_values(by='Temperature_change', ascending=False).head(10)
top10_countries.index = range(1, 11)
st.write(top10_countries)
st.sidebar.header("Top 10 Countries with Highest Temperature Anomalies from 2003-2022")
st.sidebar.info("This section highlights the top 10 countries with the highest average temperature and its trend anomalies from 2003 to 2022. Use the buttons to visualize the data in different chart formats.")
st.subheader("Top 10 Countries with Highest Temperature and Temperature Trend Anomalies from 2003-2022")

# Step 4: Bar plot

if st.button("Show Bar Chart"):
    st.subheader("Bar Chart: Top 10 Countries with Highest Temperature Anomalies (2003–2022)")
    plt.figure(figsize=(12, 8))
    plt.bar(top10_countries['Country'], top10_countries['Temperature_change'], color='skyblue')
    plt.xlabel('Country')
    plt.ylabel('Average Temperature Anomaly (°C)')
    plt.title('Top 10 Countries with Highest Temperature Anomalies (2003-2022)')
    plt.xticks(rotation=45, ha='right')  # Rotate labels for readability
    plt.legend(labels=['Top 10 Countries'], title="Legend", loc='upper right')
    st.pyplot(plt)

import seaborn as sns
top10_data = recent_temp[recent_temp['Country'].isin(top10_countries['Country'])]
if st.button("Show Line Plot"):
    st.subheader("Line Plot: Temperature Trends of Top 10 Countries (2003–2022)")
    plt.figure(figsize=(16, 10))
    sns.lineplot(data=top10_data, x='Year', y='Temperature_change', hue='Country', marker='o')
    plt.xlabel('Year')
    plt.ylabel('Average Temperature Change (°C)')
    plt.title('Top 10 Countries with Highest Average Temperature Change (2003-2022)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)
st.sidebar.header("Correlation Analysis between CO2 and Temperature Change")
st.subheader("Correlation Analysis between CO2 and Temperature Change")
# Correlation
#Compute the correlation coefficient between global CO₂ and global temperature anomaly.
annual_co2 = annual_CO2.copy()
global_temp = global_temp_change.copy()
# Merge CO₂ and temperature data on 'Year'
combined_df = pd.merge(annual_co2, global_temp_change, on='Year', how='inner')

# Calculate correlation
correlation = combined_df['CO2_ppm'].corr(combined_df['Temperature_change'])
st.write(f"Correlation between CO₂ and Temperature Anomaly: {correlation:.3f}")
st.write("A correlation coefficient close to +1 indicates a strong positive relationship, meaning as CO₂ levels rise, temperature anomalies also tend to increase. This supports the understanding that increased greenhouse gas emissions contribute to global warming.")
st.sidebar.info("This section analyzes the correlation between CO2 concentrations and global temperature anomalies. A higher correlation indicates a stronger relationship between the two variables.")

st.subheader("Highlighting years where CO₂ or temperature spiked or dropped sharply (detect anomalies)")
# Calculate year-over-year changes
combined_df['CO2_Change'] = combined_df['CO2_ppm'].diff()
combined_df['Temp_Change'] = combined_df['Temperature_change'].diff()
st.write(combined_df.head())
# Define thresholds (mean ± 2*std)
co2_thresh = combined_df['CO2_Change'].std() * 2
temp_thresh = combined_df['Temp_Change'].std() * 2
# Flag anomalies
anomalies = combined_df[
    (combined_df['CO2_Change'].abs() > co2_thresh) |
    (combined_df['Temp_Change'].abs() > temp_thresh)
]
anomalies = anomalies.set_index("Year")
Anomalies_data = st.sidebar.slider("Number of Anomalies Data:", min_value=1, max_value=len(anomalies), value=3, step=1)
st.write(f"Displaying {Anomalies_data} anomalies data.")
st.subheader("Anomalies Data Table")
anomalies = anomalies.head(Anomalies_data)
st.write(anomalies)
st.write("The anomalies identified in the dataset highlight years where there were significant deviations in CO₂ levels or temperature changes. These spikes or drops could be linked to major global events, such as economic recessions, natural disasters, or policy changes impacting emissions.")
st.sidebar.info("This section identifies and highlights years with significant anomalies in CO2 levels or temperature changes. Use the slider to adjust how many anomalies to display.")

# Plot anomalies
st.subheader("Temperature Anomalies with Outliers Highlighted")
plt.figure(figsize=(14, 6))
sns.lineplot(data=combined_df, x='Year', y='Temperature_change', label='Temp')
sns.scatterplot(data=anomalies, x='Year', y='Temperature_change', color='red', s=100, label='Anomaly')
plt.title('Global Temperature Anomalies with Outliers Highlighted')
plt.ylabel('Temperature Change (°C)')
plt.xlabel('Year')
plt.legend()
plt.tight_layout()
st.pyplot(plt)

#plot for CO2 anomalies
st.subheader("CO2 Anomalies with Outliers Highlighted")
plt.figure(figsize=(14, 6))
sns.lineplot(data=combined_df, x='Year', y='CO2_Change', label='Temp')
sns.scatterplot(data=anomalies, x='Year', y='CO2_Change', color='red', s=100, label='Anomaly')
plt.title('Global CO2 Anomalies with Outliers Highlighted')
plt.ylabel('CO2 Change (ppm)')
plt.xlabel('Year')
plt.legend()
plt.tight_layout()
st.pyplot(plt)
st.write("The highlighted anomalies in CO₂ changes indicate years with significant deviations from the norm. These outliers may correspond to extraordinary events or shifts in global emissions, providing insights into the factors influencing atmospheric CO₂ levels.")
anomalies = anomalies.reset_index()
st.write("Detected Anomalies:")
st.write(anomalies[['Year', 'CO2_Change', 'Temp_Change']])
#Mark an event that  could explain an anomaly
st.write("Notable Event: The significant dip in CO₂ levels around 2020 likely corresponds to the global economic slowdown caused by the COVID-19 pandemic, which led to reduced industrial activity and transportation emissions.")
plt.axvline(x=2020, linestyle='--', color='gray')
plt.text(2020 + 0.2, combined_df['Temperature_change'].max() - 0.1, 
         'COVID-19 Lockdown', rotation=90)
st.pyplot(plt)


st.sidebar.header("Simulating 'What-If' Scenarios")
st.subheader("Simulating 'What-If' Scenarios: Impact of Reducing CO₂ Emissions by 20%")
# Simulate reducing CO2 by 20%
reduced_co2 = combined_df.copy()
reduced_co2['CO2_ppm'] = reduced_co2['CO2_ppm'] * 0.8
# Simple linear regression to estimate temperature change based on CO2
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X = combined_df[['CO2_ppm']]
y = combined_df['Temperature_change']
model.fit(X, y)
predicted_temp = model.predict(reduced_co2[['CO2_ppm']])
reduced_co2['Predicted_Temp_Change'] = predicted_temp
st.write(reduced_co2.head())
st.write("By simulating a 20% reduction in CO₂ emissions, we can estimate the potential impact on global temperature anomalies. The linear regression model provides a simplified view of how temperature changes might respond to lower CO₂ levels.")
st.sidebar.info("This section simulates a 'what-if' scenario where CO2 emissions are reduced by 20%. It estimates the potential impact on global temperature anomalies using a linear regression model.")
st.subheader("Impact of 20% CO₂ Reduction on Global Temperature Anomalies")
plt.figure(figsize=(12, 6)) 
plt.plot(combined_df['Year'], combined_df['Temperature_change'], 'r-o', label='Actual Temp Change')
plt.plot(reduced_co2['Year'], reduced_co2['Predicted_Temp_Change'], 'b--s', label='Predicted Temp Change (20% CO₂ Reduction)')
plt.title('Impact of 20% CO₂ Reduction on Global Temperature Anomalies')
plt.ylabel('Temperature Change (°C)')
plt.xlabel('Year')
plt.legend()
plt.grid(alpha=0.3)
st.pyplot(plt)
st.write("The graph compares actual global temperature changes with predicted changes under a scenario of 20% reduced CO₂ emissions. The predicted trend suggests that lowering CO₂ levels could mitigate some of the temperature increases, highlighting the importance of emission reduction strategies in combating climate change.")

#Use a simple linear regression model to predict global temperature anomaly based on CO₂ concentration.
st.sidebar.header("Linear Regression Model: Predicting Temperature Change from CO₂ Levels")
st.subheader("Linear Regression Model: Predicting Temperature Change from CO₂ Levels")
# Prepare features (CO2) and target (Temperature)
X = combined_df[['CO2_ppm']]
y = combined_df['Temperature_change']

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Print coefficients
slope = model.coef_[0]
intercept = model.intercept_
st.write(f"Model: Temperature = {slope:.4f} * CO₂ + {intercept:.4f}")
st.write("The linear regression model quantifies the relationship between CO₂ concentrations and global temperature anomalies. The slope indicates the expected change in temperature for each unit increase in CO₂ levels, while the intercept represents the baseline temperature anomaly when CO₂ is zero.")
st.write("This model can be used to predict future temperature changes based on projected CO₂ levels, aiding in climate impact assessments and policy planning.")
# Predict temperature for a given CO2 level
st.subheader("Predicting 10% and 20% Temperature Change for a Given CO₂ Level in the next 10 years (2022-2034)")
st.sidebar.info("This section uses the linear regression model to predict future temperature changes based on different CO2 scenarios over the next decade (2022-2034). It illustrates how varying CO2 levels can impact global temperatures by 10% increase and 10% decrease.")
# Get the latest CO2 value (2022)
latest_co2 = annual_co2.loc[annual_co2['Year'] == 2022, 'CO2_ppm'].values[0]

# Simulated years
future_years = np.arange(2022, 2035)

# Scenarios
scenarios = {
    "Current": latest_co2 * np.ones(len(future_years)),
    "10% Increase": latest_co2 * 1.10 * np.ones(len(future_years)),
    "10% Decrease": latest_co2 * 0.90 * np.ones(len(future_years)),
}

# Predict temperatures
predictions = {}
for label, co2_values in scenarios.items():
    predictions[label] = model.predict(co2_values.reshape(-1, 1))

plt.figure(figsize=(12, 6))

# Loop through scenarios and plot with shading
for label, temps in predictions.items():
    temp_df = pd.DataFrame({'Year': future_years, 'Temperature': temps})
    
    # Line for the scenario
    sns.lineplot(data=temp_df, x='Year', y='Temperature', label=label)
    
    # Add a little shading around each line (±0.05°C just as illustrative uncertainty)
    plt.fill_between(
        temp_df['Year'], 
        temp_df['Temperature'] - 0.05, 
        temp_df['Temperature'] + 0.05, 
        alpha=0.1
    )

plt.title('Projected Global Temperature Under Different CO₂ Scenarios (2022–2034)')
plt.ylabel('Predicted Temperature Change (°C)')
plt.xlabel('Year')
plt.legend(title="Scenario")
plt.grid(alpha=0.3)

# Streamlit rendering
st.pyplot(plt)
st.write("The projections illustrate how different CO₂ scenarios could influence global temperature anomalies over the next decade. An increase in CO₂ levels is associated with higher temperature anomalies, while a significant reduction could help stabilize or even lower temperatures. This underscores the critical role of emission control in shaping future climate outcomes.")


st.subheader("Conclusions")
st.write("""Based on the data analysis and visualizations presented in this app, several key conclusions can be drawn regarding the relationship between CO₂ emissions and global temperature trends:

1. **Rising CO₂ Levels**: The analysis of atmospheric CO₂ concentrations from 1958 to 2023 reveals a consistent upward trend, indicating that human activities continue to contribute significantly to greenhouse gas emissions.
2. **Temperature Anomalies**: The global average temperature change data from 1960 to 2022 shows a clear warming trend, with temperature anomalies increasing over time. This trend is consistent with the scientific consensus on global warming.
3. **Strong Correlation**: The correlation analysis between CO₂ levels and temperature anomalies indicates a strong positive relationship, suggesting that increases in CO₂ concentrations are closely linked to rising global temperatures.
4. **Impact of Anomalies**: The identification of anomalies in CO₂ and temperature data highlights the influence of extraordinary events, such as the COVID-19 pandemic, which temporarily reduced emissions and affected temperature trends.
5. **What-If Scenarios**: Simulations of a 20% reduction in CO₂ emissions suggest that such measures could mitigate some of the temperature increases, emphasizing the potential benefits of aggressive emission reduction strategies.
6. **Predictive Modeling**: The linear regression model demonstrates the ability to predict future temperature changes based on CO₂ levels, providing a useful tool for assessing the potential impacts of different emission scenarios.
7. **Future Projections**: The projections for the next decade (2022-2034) under different CO₂ scenarios illustrate the significant influence that emission levels can have on global temperature trends, reinforcing the urgency of climate action.
These conclusions underscore the critical need for continued efforts to reduce CO₂ emissions and implement effective climate policies to mitigate the impacts of global warming. The data-driven insights provided by this app can inform decision-making and promote awareness of the challenges and opportunities associated with climate change."""
)


st.subheader("Recommendations")
st.write("""Based on the analysis presented in this app, several key recommendations emerge for addressing climate change:
1. **Aggressive Emission Reductions**: Implement policies to significantly reduce CO₂ emissions, aiming for at least a 20% reduction in the near term. This could involve transitioning to renewable energy sources, enhancing energy efficiency, and promoting sustainable transportation.
2. **International Collaboration**: Climate change is a global issue requiring coordinated international efforts. Countries should work together to set ambitious targets, share technology, and provide financial support to developing nations for climate mitigation and adaptation.
3. **Investment in Research and Innovation**: Support research into new technologies for carbon capture, renewable energy, and climate resilience. Innovation can drive down costs and improve the effectiveness of climate solutions.
4. **Public Awareness and Education**: Increase public understanding of climate change through education campaigns. Informed citizens are more likely to support and engage in sustainable practices and policies.
5. **Adaptation Strategies**: Develop and implement strategies to adapt to the impacts of climate change that are already occurring, such as rising sea levels
and increased frequency of extreme weather events. This includes investing in resilient infrastructure and protecting vulnerable ecosystems.
6. **Regular Monitoring and Reporting**: Establish robust systems for monitoring CO₂ levels and temperature changes, ensuring transparency and accountability in climate actions. Regular reporting can help track progress and adjust strategies as needed.
By following these recommendations, policymakers, businesses, and individuals can contribute to mitigating climate change and fostering a sustainable future for all.
""")
thank_you_note = st.text_input("Any suggestions or feedback for improving this app?", "Your feedback is valuable!")
st.write(f"Thank you for your feedback: {thank_you_note}")

st.subheader("Thank you for using the Tracking Climate Change App. Together, we can make a difference in addressing climate change through informed actions and policies.")
