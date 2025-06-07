# Databricks notebook source
# MAGIC %md
# MAGIC # 2.3 EDA and Visualization

# COMMAND ----------


# Load the data
df = spark.table("assignment2.default.voterfile_export")
df.createOrReplaceTempView("AK_table")
display(df)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Use 5 different Pyspark Pandas APIs to perform various data exploration tasks.

# COMMAND ----------

# Use 5 different Pyspark Pandas APIs to perform various data exploration tasks.

# 1. Shows the structure of DataFrame — column names, types, and nullability.
df.printSchema()

# 2. Select specific columns and view unique values (great for exploring categorical variables).
df.select("Voters_Active").distinct().show()


# 3. Group rows by a column and count how many fall into each category (e.g., how many active vs inactive voters).
df.groupBy("Voters_Active").count().show()


# 4. Filter the rows based on a active status.
display(df.filter(df["Voters_Active"] == "A"))


# 5. Summary statistics for numeric columns — count, mean, stddev, min, and max.
df.select("Voters_Age", "CommercialData_EstimatedHHIncomeAmount").describe().show()



# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize voter turnout by age groups: 20s, 30s, ..., 80+. Each age groups is to be plotted as a line trace in a shared plot.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col, when

# Create age group buckets based on Voters_Age
df_grouped = df.withColumn("AgeGroup", when((col("Voters_Age") >= 20) & (col("Voters_Age") < 30), "20s")
                                     .when((col("Voters_Age") >= 30) & (col("Voters_Age") < 40), "30s")
                                     .when((col("Voters_Age") >= 40) & (col("Voters_Age") < 50), "40s")
                                     .when((col("Voters_Age") >= 50) & (col("Voters_Age") < 60), "50s")
                                     .when((col("Voters_Age") >= 60) & (col("Voters_Age") < 70), "60s")
                                     .when((col("Voters_Age") >= 70) & (col("Voters_Age") < 80), "70s")
                                     .when((col("Voters_Age") >= 80), "80+")
                                     .otherwise("Unknown"))



# COMMAND ----------

# Filter Active Voters and Count by AgeGroup

df_active = df_grouped.filter(col("Voters_Active") == "A")
df_turnout = df_active.groupBy("AgeGroup").count()


# COMMAND ----------

import matplotlib.pyplot as plt

# Convert to Pandas and sort
pd_df = df_turnout.toPandas().sort_values("AgeGroup")

# Plot line chart
plt.plot(pd_df["AgeGroup"], pd_df["count"], marker="o")
plt.title("Voter Turnout by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Active Voter Count")
plt.grid(True)
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a choropleth visualization of voter turnout rate by county for a chosen year.

# COMMAND ----------

# MAGIC %md
# MAGIC Check the turnout rate by county in Alaska using a table

# COMMAND ----------

from pyspark.sql.functions import regexp_replace, col

# Remove % and convert to float
df_cleaned = df.withColumn(
    "TurnoutRate",
    regexp_replace(col("ElectionReturns_G08CountyTurnoutAllRegisteredVoters"), "%", "").cast("float")
)

# Group by County to get unique values
df_turnout_by_county = df_cleaned.select("County", "TurnoutRate").distinct()

# Display the table
display(df_turnout_by_county)


# COMMAND ----------

# MAGIC %md
# MAGIC Generate a choropleth map to see the distribution of counties in Alaska

# COMMAND ----------

import geopandas as gpd
import plotly.express as px

# Load Alaska counties (or CA if you're using California FIPS)
alaska_gdf = gpd.read_file("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json")

# Filter for Alaska counties (FIPS code 02)
alaska_gdf = alaska_gdf[alaska_gdf["STATE"] == "02"]

# Add sample data directly to the GeoDataFrame
alaska_gdf["value"] = range(len(alaska_gdf))

fig = px.choropleth_mapbox(
    alaska_gdf,  # Use alaska_gdf instead of gdf
    geojson=alaska_gdf.geometry.__geo_interface__,
    locations=alaska_gdf.index,
    color="value",
    mapbox_style="carto-positron",
    center={"lat": 64.2, "lon": -149.5},
    zoom=3.5,
    opacity=0.7,
    title="Alaska County Demo Choropleth"
)

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Generate the choropleth map of the turnout rate of each county in Alaska

# COMMAND ----------

import geopandas as gpd
import plotly.express as px

# Load Alaska counties
alaska_gdf = gpd.read_file("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json")
alaska_gdf = alaska_gdf[alaska_gdf["STATE"] == "02"]

# Clean up and normalize county names
alaska_gdf["County"] = alaska_gdf["NAME"].str.upper()
df_turnout_by_county_pd = df_turnout_by_county.toPandas()
df_turnout_by_county_pd["County"] = df_turnout_by_county_pd["County"].str.upper()

# Merge turnout data into alaska_gdf
alaska_merged = alaska_gdf.merge(df_turnout_by_county_pd, on="County", how="left")

fig = px.choropleth_mapbox(
    alaska_merged,
    geojson=alaska_merged.geometry.__geo_interface__,
    locations=alaska_merged.index,
    color="TurnoutRate",
    color_continuous_scale="Blues",
    mapbox_style="carto-positron",
    center={"lat": 64.2, "lon": -149.5},
    zoom=3.5,
    opacity=0.75,
    title="Alaska Voter Turnout Rate by County (2008)"
)

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a choropleth visualization of party affiliation by county

# COMMAND ----------

# MAGIC %md
# MAGIC Create a new column DominantParty based on highest turnout rate

# COMMAND ----------

from pyspark.sql.functions import regexp_replace, col, greatest, when

# Clean percentages → floats
df_cleaned = df.withColumn("Dem",
    regexp_replace(col("ElectionReturns_G08CountyTurnoutDemocrats"), "%", "").cast("float")
).withColumn("Rep",
    regexp_replace(col("ElectionReturns_G08CountyTurnoutRepublicans"), "%", "").cast("float")
).withColumn("Ind",
    regexp_replace(col("ElectionReturns_G08CountyTurnoutIndependentsAllOthers"), "%", "").cast("float")
)

# Determine dominant party
df_party = df_cleaned.withColumn(
    "DominantParty",
    when((col("Dem") > col("Rep")) & (col("Dem") > col("Ind")), "DEMOCRAT")
    .when((col("Rep") > col("Dem")) & (col("Rep") > col("Ind")), "REPUBLICAN")
    .otherwise("INDEPENDENT")
)

# Select one row per county
df_party_by_county = df_party.select("County", "DominantParty").distinct()


# COMMAND ----------

# MAGIC %md
# MAGIC Convert to Pandas and Normalize

# COMMAND ----------

df_party_pd = df_party_by_county.toPandas()
df_party_pd["County"] = df_party_pd["County"].str.upper()


# COMMAND ----------

# MAGIC %md
# MAGIC  Merge with Alaska GeoDataFrame

# COMMAND ----------

import geopandas as gpd

# Load Alaska counties shapefile
alaska_gdf = gpd.read_file("https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json")
alaska_gdf = alaska_gdf[alaska_gdf["STATE"] == "02"]
alaska_gdf["County"] = alaska_gdf["NAME"].str.upper()

# Merge with party data
alaska_party_map = alaska_gdf.merge(df_party_pd, on="County", how="left")


# COMMAND ----------

# MAGIC %md
# MAGIC Plot Choropleth

# COMMAND ----------

import plotly.express as px

fig = px.choropleth_mapbox(
    alaska_party_map,
    geojson=alaska_party_map.geometry.__geo_interface__,
    locations=alaska_party_map.index,
    color="DominantParty",
    color_discrete_map={
        "DEMOCRAT": "blue",
        "REPUBLICAN": "red",
        "INDEPENDENT": "gray"
    },
    mapbox_style="carto-positron",
    center={"lat": 64.2, "lon": -149.5},
    zoom=3.5,
    opacity=0.75,
    title="Dominant Party by County (2008 Turnout)"
)

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.show()
