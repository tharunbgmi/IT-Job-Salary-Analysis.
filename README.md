# Indian IT Job Salary Analysis

This project explores factors affecting IT job salaries in India using a dataset containing information on Experience, Job Title, Location, Education, Age, Gender, and Salary. The analysis covers data cleaning, exploratory data analysis, statistical testing, and building a multiple linear regression model to quantify key factors influencing salary.

## Objectives

- Understand how job title, experience, education, location, age, and gender affect salary.
- Test the significance of salary differences between job titles.
- Build a regression model to predict salary from multiple variables.
- Document insights and visualize results for clarity.

## Dataset Description

The dataset contains 1000 records with these columns:

- **Education**: Highest education level (High School, Bachelor, Master, PhD)
- **Experience**: Years of professional experience
- **Location**: Location type (Urban, Suburban, Rural)
- **Job_Title**: Job role (Analyst, Engineer, Manager, Director)
- **Age**: Age in years
- **Gender**: Male or Female
- **Salary**: Annual salary in currency units

## Data Cleaning and Preparation

- Checked for and handled missing or duplicated data.
- Converted categorical columns (`Job_Title`, `Location`, `Education`, `Gender`) into numeric dummy variables for regression.
- Ensured all features used in the model are numeric and properly formatted.

## Exploratory Data Analysis (EDA)

- Calculated average salaries grouped by job title, education, location, and gender.
- Visualized salary distributions using boxplots and scatterplots.
- Found that salary varies notably with job title, education, location, and experience.

## Statistical Testing

- Performed one-way ANOVA testing on salary by job title: showed statistically significant differences.
- Post-hoc Tukey’s HSD test identified which job title pairs differ significantly in mean salary.

## Regression Modeling

- Built a multiple linear regression model to predict salary based on experience, job title, location, education, age, and gender.
- Key findings from model coefficients:
  - Experience increases salary by ~1031 units per year.
  - Directors earn ~25,380 units more than Analysts (baseline).
  - Urban and Suburban locations have significantly higher salaries than Rural.
  - Education has strong effects; PhD holders earn ~40,000 units more than Bachelors.
  - Gender and Age were not statistically significant factors.
- Model R-squared of 0.878 indicates strong explanatory power.

## Residual Analysis

- Residual plots show no obvious violation of linear regression assumptions.
- Residuals are roughly normally distributed and variance appears constant across predicted values.

## Summary of Key Findings

- Experience, job title, location, and education strongly influence salary levels.
- Gender and age do not significantly affect salaries in this dataset.
- The model successfully explains a high proportion of salary variability.

## How to Run

- Install required Python packages: `pandas`, `seaborn`, `statsmodels`, `matplotlib`.
- Open Jupyter Notebook `salary_analysis.ipynb`.
- Run cells sequentially to reproduce analysis, visualizations, and final model.

