import numpy as np
import pickle
import pandas as pd
import streamlit as st
from transformers import pipeline
from PIL import Image
from model_file import TapasInference
import time

tqa = pipeline(task="table-question-answering", model="google/tapas-base-finetuned-wtq")
obj = 1
def main():
    global tqa
    global obj
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">TabQ: Ask natural query from tabular data </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    option = st.selectbox('Select Model: ',('tapas-base-finetuned-wtq', 'none'))
    # st.write('Chosen model: ', option)
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)
    # col1, col2, col3, col4, col5, col6 = st.columns(6)

    # if col6.button("Load"):
    #     if len(df) > 100:
    #         st.error('Please choose a smaller table')
    #     elif len(df) < 5:
    #         st.error('Please choose a bigger table')
    #     else:
    #         print(obj)
    #         # obj = TapasInference(option, df)
    #         # if obj.flag:
    #         #     st.success('Model loaded successfully')
    #         #     print('inside', obj.flag)
    #         else:
    #             st.error('Model loading error!!!')

    query = st.text_input('Enter your query here...')
    index_start_time = time.time()
    if query:
        obj = TapasInference(tqa, df)
        if obj:
            res = obj.infer(query)
            # res = tqa(table=df.astype(str), query=query)
            st.text("Answer:")
            if res:
                st.info(res)
            else:
                st.info("Please try another query!!!")
        else:
            st.text("Model not found!")
    else:
        st.text("No query found!")
    index_end_time = time.time()
    duration = index_end_time - index_start_time
    st.text("time taken to query: "+str(duration)+" sec")
    print(obj)


if __name__ == '__main__':
    main()