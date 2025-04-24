
import os
import io
import base64
import streamlit as st
from PIL import Image
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
# ------ Configuration ------
# API keys should be set as environment variables for security
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# GENAI_API_KEY = os.getenv('GENAI_API_KEY')
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GENAI_API_KEY = st.secrets["GENAI_API_KEY"]


# Initialize clients
oai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GENAI_API_KEY)

# Directory paths
REF_DIR = 'reference_images'
OUT_DIR = 'outputs'

# Clear reference images on each run
if os.path.exists(REF_DIR):
    for f in os.listdir(REF_DIR):
        os.remove(os.path.join(REF_DIR, f))

os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ------ Gemini Helper Functions ------
def setup_gemini():
    """
    Ensures Gemini API is configured.
    """
    try:
        genai.configure(api_key=GENAI_API_KEY)
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}")
        return False


def generate_text_prompt(post_text: str, model_name: str = 'gemini-2.0-flash') -> str:
    """
    Generates a creative image prompt from the given post text using Gemini.
    """
    if not setup_gemini():
        return ''
    model = genai.GenerativeModel(model_name)
    try:
        prompt=f"""
# Overview
You are an AI agent that transforms LinkedIn posts into visual prompt descriptions for generating graphic marketing materials.
 These visuals are designed to be paired with the post on LinkedIn, helping communicate the message in a visually engaging, brand-aligned way.
## Objective:
- Read and analyze the given LinkedIn post.
- Identify the main message, insight, or takeaway from the post.
- Create a clear and compelling graphic prompt that can be used with a text-to-image generator.
- The result should be a marketing-style graphic, not a literal scene or hyperrealistic photo,
that:
1) Visually supports or illustrates the key idea of the post 
2) Looks appropriate for use in a professional LinkedIn feed 
3) Feels polished, modern, and engaging

## Output Instructions:
- Output only the final image prompt. Do not output quotation marks.
- Do not repeat or rephrase the LinkedIn post.
- Do not add any explanations or extra content just the image prompt.
- Never leave things blank like "Header area reserved for customizable callout text"
- Output numeric stats when available in the original post
## Style Guidelines:
- Think like a brand designer or marketing creative.
- Visuals may include: text, charts, icons, abstract shapes, overlays, modern illustrations, motion-like effects, 
bold typography elements (described, not rendered), or metaphorical concepts.
You can mention layout suggestions (e.g., "split screen design," "header with bold title and subtle background illustration"). 
Assume the output will be generated using AI image tools - your prompt should guide those tools effectively.
## Example Prompt Format:
A modern flat-style graphic showing a human brain connected to mechanical gears, representing the fusion of AI and automation. 
Minimalist background, soft gradients, clean sans-serif text placement space at the top 

Give LinkedIn post: '{post_text}'"""
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else ''
    except Exception as e:
        st.error(f"Gemini generation failed: {e}")
        return ''

# ------ Streamlit App ------
st.set_page_config(page_title='Brand-Based Visual Generator', layout='wide')
st.title('🔮 Brand-Based Visual Generator')

# ==== 1) Reference Image Management ====
st.header('1. Upload or Manage Reference Images')
# Upload
uploaded_files = st.file_uploader(
    'Upload reference images (JPEG/PNG)',
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True
)
if uploaded_files:
    for up in uploaded_files:
        path = os.path.join(REF_DIR, up.name)
        if not os.path.exists(path):
            with open(path, 'wb') as f:
                f.write(up.getbuffer())
            st.success(f"Saved {up.name}")

# Display existing references
st.subheader('Saved Reference Images')
refs = os.listdir(REF_DIR)
cols = st.columns(4)
for idx, fname in enumerate(refs):
    img_path = os.path.join(REF_DIR, fname)
    with cols[idx % 4]:
        st.image(img_path, width=100, caption=fname)
        if st.button(f'Delete {fname}', key=f'del_{fname}'):
            os.remove(img_path)
            st.rerun()

# ==== 2) Post Input & Image Generation ====
st.header('2. Generate Image for Your Post')
post_text = st.text_area('Enter your post text here', height=150)

# ------ CHANGED: display previously generated image from session state ------
if 'last_image_bytes' in st.session_state:
    st.subheader('Previously Generated Image')
    st.image(st.session_state['last_image_bytes'], use_container_width=True)

if st.button('Generate Image'):
    has_refs = len(os.listdir(REF_DIR)) > 0
    proceed = True
    if not has_refs:
        st.warning('No reference images found. Continue without a brand guide?')
        proceed = st.button('Yes, continue without images')
    if proceed:
        if not post_text:
            st.error('Please enter some post text.')
        else:
            # 2a) Create prompt via Gemini
            with st.spinner('Generating image prompt...'):
                image_prompt = generate_text_prompt(post_text)
            if not image_prompt:
                st.error('Failed to generate image prompt.')
            else:
                image_prompt += " Use all the reference pictures as a reference to inform the style of the post generated."
                st.markdown(f"**Generated Prompt:** {image_prompt}")

                # 2b) Use all reference images for edit
                ref_paths = [os.path.join(REF_DIR, fn) for fn in os.listdir(REF_DIR)]
                files = [open(p, 'rb') for p in ref_paths]
                try:
                    with st.spinner('Generating image...'):
                        result = oai_client.images.edit(
                            model='gpt-image-1',
                            image=files,
                            prompt=image_prompt
                        )
                    b64 = result.data[0].b64_json
                    img_bytes = base64.b64decode(b64)
                    img = Image.open(io.BytesIO(img_bytes))

                    # ------ CHANGED: auto-save generated image to outputs folder ------
                    out_name = f"generated_{len(os.listdir(OUT_DIR))+1}.png"
                    out_path = os.path.join(OUT_DIR, out_name)
                    with open(out_path, 'wb') as f:
                        f.write(img_bytes)
                    # store in session state
                    st.session_state['last_image_bytes'] = img_bytes
                    st.session_state['last_image_name'] = out_name

                    # Display
                    st.subheader('Generated Image')
                    st.image(img_bytes, use_container_width=True)

                    # ------ CHANGED: use download_button instead of save button ------
                    st.download_button(
                        label='Download Generated Image',
                        data=img_bytes,
                        file_name=out_name,
                        mime='image/png'
                    )

                except Exception as e:
                    st.error(f"Image generation failed: {e}")
                finally:
                    for f in files:
                        f.close()
