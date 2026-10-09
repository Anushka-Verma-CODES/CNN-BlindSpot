import streamlit as st
import torch

from data.datasets import get_train_val_loaders
from models.network import SimpleMLP
from models.trainer import train_model
from visualization.curves import plot_loss_curve, plot_accuracy_curve

st.set_page_config(page_title = "NeuroFix", layout = "wide")
st.title("NeuroFix")

if st.button("Run Baseline Experiment"):

    with st.spinner("Training Model..."):

        t_loader, v_loader = get_train_val_loaders(batch_size=32)

        model = SimpleMLP()

        history = train_model(model, t_loader, v_loader, epochs = 150)

        loss_fig = plot_loss_curve(history)
        acc_fig = plot_accuracy_curve(history)

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(loss_fig, use_container_width=True)
        with col2:
            st.plotly_chart(acc_fig, use_container_width=True)

        st.success("Experiment Done")