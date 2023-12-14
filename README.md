# <center>Health Indicators and their relationshp to having Diabetes</center>
<center>By: Casey Cruz</center> 

---

## Introduction
> Diabetes is a chronic medical condition that affects how the body creates energy from food. The food consumed is broken down into sugar which is released into the bloodstream. When this happens,  blood sugar levels go up, but the pancreas releases insulin to stabilize it. If a person has diabetes, their body cannot regulate blood sugar levels through the use of insulin as well as someone without diabetes. This results in too much blood sugar remaining in the bloodstream and can cause serious health concerns.
  
> In the US there are around 38 million adults that have diabetes. It is the 8th leading cause of death in the US and the number 1 cause of kidney failure, lower-limb amputation, and adult blindness. The number of adults that have been diagnosed with diabetes has doubled in the last twenty years which is a major problem. For this project, the dataset that will be used is from the Behavioral Risk Factor Surveillance System (BRFSS) which collects data on health-related risk behaviors and chronic health conditions. The specific dataset that will be analyzed contains data on whether a person has no diabetes or has diabetes/prediabetes and health indicators that they have or do not have. Examples of the indicators are high blood pressure and high cholestrol. This data can be used to understand the health indicators that relate to someone having diabetes. With that, information on general preventative measures can be identified and released to the public to show them which things are important to reduce their chances of having diabetes.

## How to use the Data Science Pipeline to Analyze Data
> When working with data, there are four main aspects that are used. They are __<font color = green>data collection</font>__, __<font color = green>data storage</font>__, __<font color = green>data analyis</font>__, and __<font color = green>data visualization</font>__. Data Collection is the process of gathering raw information from various sources. This can be done through web scraping. Data Storage is organizing, cleaning, and storing the data in a structured format that allows for easy retrieval and analysis. Data Analysis is where stored data is transformed and manipulated to extract meaningful insights. Data Presentation is communicating the results of data analysis in a clear and understandable manner. This often done through visualizations.

---

### Objective
>In this project the overall objective will be to create a classification algorithm to predict if their is a presense of diabetes based on health indicators. Python will be the programming language used.

---

## <font color = green>Data Collection</font>
> The data that will be analysing has already been collected. However, it is important to understand data collection since it is the first step in the data science pipeline so an example will be run through.

>Data collection is the process of gathering raw information from sources and can perform this by web scraping. Web scraping is the process of extracting information or data from websites. In the example well will be extracting data from https://www.cnn.com.

>Here we can see the steps of web scraping. The first step is to make a request to the site to access and then parse the HTML code of the site. We extract the article references and then request and parse their HTML code with BeautifulSoup. The article title gets extracted and printed. The data collected can then be saved, processed, or used for analysis. <font color = red>Follow along with comments in the code!</font>
