import streamlit as st
import torch
from PIL import Image
import os
from torchvision.transforms import ToPILImage
import time
from model import StyleTransferModel
import io


def get_checkpoint_path(content_name, style_name):
    """Generate a unique checkpoint path based on content and style image names"""
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"{content_name}_{style_name}_checkpoint.pth")


def main():
    st.set_page_config(page_title="Neural Style Transfer", layout="wide")
    st.title("Neural Style Transfer App")
    st.write("Upload a content image and a style image to create an artistic transformation!")

    @st.cache_resource
    def load_model():
        return StyleTransferModel()

    model = load_model()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Content Image")
        content_file = st.file_uploader("Choose your content image", type=['png', 'jpg', 'jpeg'])
        if content_file is not None:
            content_image = Image.open(content_file)
            st.image(content_image, caption='Content Image', use_column_width=True)

    with col2:
        st.subheader("Style Image")
        style_file = st.file_uploader("Choose your style image", type=['png', 'jpg', 'jpeg'])
        if style_file is not None:
            style_image = Image.open(style_file)
            st.image(style_image, caption='Style Image', use_column_width=True)

    with st.expander("Advanced Settings"):
        num_steps = st.slider("Number of optimization steps", min_value=100, max_value=2000, value=500, step=100)
        use_checkpoint = st.checkbox("Use cached results if available", value=True)

    if st.button('Generate Stylized Image'):
        if content_file is not None and style_file is not None:
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()

                checkpoint_path = get_checkpoint_path(
                    content_file.name,
                    style_file.name
                ) if use_checkpoint else None

                status_text.text('Starting style transfer...')
                start_time = time.time()

                result_tensor = model.transform_image(
                    content_file,
                    style_file,
                    checkpoint_path if use_checkpoint else None
                )

                result_image = ToPILImage()(result_tensor.squeeze(0).cpu())
                end_time = time.time()
                processing_time = end_time - start_time
                st.success(f'Style transfer completed in {processing_time:.2f} seconds!')

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.subheader("Stylized Result")
                    st.image(result_image, caption='Stylized Image', use_column_width=True)

                    if result_image is not None:
                        buf = io.BytesIO()
                        result_image.save(buf, format='PNG')
                        st.download_button(
                            label="Download Stylized Image",
                            data=buf.getvalue(),
                            file_name="stylized_image.png",
                            mime="image/png"
                        )

            except Exception as e:
                st.error(f'An error occurred: {str(e)}')

        else:
            st.warning('Please upload both content and style images.')


if __name__ == "__main__":
    main()