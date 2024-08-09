import streamlit as st
import requests

st.set_page_config(page_title='Simple Bird Classification', layout='wide')

FLASK_API_URL = 'http://localhost:5000/predict'

def main():
    st.title('Simple Bird Classification 🐦')
    st.write('This is a simple bird classification app that uses a ResNet18 model trained on 20 different bird types.')

    img_input = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if img_input is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.image(img_input, caption='Uploaded Image.', use_column_width=True)

        with col2:
            if st.button('Predict'):
                img_bytes = img_input.read()

                response = requests.post(
                    FLASK_API_URL,
                    files={'file': ('image.jpg', img_bytes, 'image/jpeg')}
                ) # nosec

                if response.status_code == 200:
                    result = response.json()
                    st.write(f'**Predicted bird type:** {result["predicted_bird_type"]}')

                    for bird_name, prob in result['probabilities'].items():
                        st.write(f"**{bird_name}:** {prob:.4f}")
                        st.progress(prob)
                else:
                    st.error(f'Error: {response.json().get("error", "Unknown error")}')
    
if __name__ == "__main__":
    main()
