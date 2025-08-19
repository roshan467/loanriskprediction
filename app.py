import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your cleaned dataset (replace with your actual data source)
# For demo, let's create dummy data (replace with your df)
# df = pd.read_csv('your_data.csv')

# Example dummy data (remove this in your real code)
df = pd.DataFrame({
    'LOAN_AMNT': [10000, 20000, 15000, 30000, 25000],
    'LOAN_INT_RATE': [5.0, 7.2, 6.5, 8.1, 7.0],
    'LOAN_GRADE': ['A', 'B', 'A', 'C', 'B'],
    'LOAN_STATUS': ['Fully Paid', 'Charged Off', 'Fully Paid', 'Charged Off', 'Fully Paid'],
    'LOAN_INTENT': ['PERSONAL', 'EDUCATION', 'PERSONAL', 'MEDICAL', 'PERSONAL']
})

st.title("Loan Dataset Exploratory Data Analysis")

st.header("Distribution of Loan Amount")
fig1, ax1 = plt.subplots()
sns.histplot(df['LOAN_AMNT'], bins=20, ax=ax1)
st.pyplot(fig1)

st.header("Loan Interest Rate by Loan Grade")
fig2, ax2 = plt.subplots()
sns.boxplot(x='LOAN_GRADE', y='LOAN_INT_RATE', data=df, ax=ax2)
st.pyplot(fig2)

st.header("Loan Status by Loan Intent")
fig3, ax3 = plt.subplots()
sns.countplot(x='LOAN_STATUS', hue='LOAN_INTENT', data=df, ax=ax3)
st.pyplot(fig3)
