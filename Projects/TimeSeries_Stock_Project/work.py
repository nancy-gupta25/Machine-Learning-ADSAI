import streamlit as st
import pandas as pd
name=st.text_input(label="Enter ur name", placeholder="Type ur name...........")
age=st.number_input(label="Enter ur age", placeholder="Type ur age......", value=None)

data: dict[str, int]={
    "Name":["Arjun","Sahil","Karan"],
    "Age":[19,20,21]
}
# df =pd.DataFrame(data=data)
# st.data_editor(df)
if st.button("Submit"):
    st.write(f"My name is {name}")
    st.write(f"I'm {int(age)} years old")
    data["Name"].append(name)
    data["Age"].append(age)
    df =pd.DataFrame(data=data)
    st.data_editor(df)
