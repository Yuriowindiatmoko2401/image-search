import streamlit as st
import helper_imgsearch, config
import numpy as np
from jina import Document


# function to show image and its predicted breed
def show_pet_and_breed(tags, image):
    """
    Shows an image of a pet and prints out its predicted breed and probability using the tags dictionary
    """
    breed = tags["label"]  # the predicted breed
    pet_category = (
        "cat" if breed[0].isupper() else "dog"
    )  # capitalized breed categories are cats, otherwise dogs
    breed = breed.lower()
    breed = " ".join(
        breed.split("_")
    )  # multi-word categories are joined using '_', so replace it with space
    article = (
        "an" if breed[0] in ["a", "e", "i", "o", "u"] else "a"
    )  # the definite article for category to be printed!
    st.image(
        image,
        caption="I am {} percent sure this is {} {} {}".format(
            round(tags["prob"] * 100), article, breed, pet_category
        ),
    )


# UI layout
st.set_page_config(page_title="Image search")
st.markdown(
    body=helper.UI.css,
    unsafe_allow_html=True,
)
# Sidebar
st.sidebar.markdown(helper.UI.about_block, unsafe_allow_html=True)

# Title
st.header("Jina Pet Breed Classification")

# File uploader
upload_cell, preview_cell = st.columns([12, 1])
query = upload_cell.file_uploader("Upload for Search Image")

# If file is uploaded
if query:
    # if clicked on 'classify' button
    if st.button(label="Classify"):
        # get tags (predicted breed and probability) for the given pet image
        tags, image = helper.get_breed(
            query, host=config.HOST, protocol=config.PROTOCOL, port=config.PORT
        )
        # show the image and its predicted greed!
        show_pet_and_breed(tags, image)
