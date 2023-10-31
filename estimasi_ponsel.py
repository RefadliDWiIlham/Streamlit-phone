import pickle
import streamlit as st

model = pickle.load(open('estimasi_harga_ponsel.sav','rb'))

st.title('ESTIMASI HARGA PONSEL')

Sale = st.number_input('MASUKAN JUMLAH PENJUALAN')
weight = st.number_input('MASUKAN BERAT')
ppi = st.number_input('MASUKAN PPI')
cpu_core = st.number_input('MASUKAN CPU CORE')
ram = st.number_input('MASUKAN RAM')
Front_Cam =st.number_input('MASUKAN FRONT CAM')
battery =st.number_input('MASUKAN BATTERY')
thickness =st.number_input('MASUKAN KETEBALAN')

predict = ''

if st.button('ESTIMASI'):
    predict = model.predict(
        [[Sale,weight,ppi,cpu_core,ram,Front_Cam,battery,thickness]]
    )

    st.write ('ESTIMASI HARGA PONSEL:', predict)
