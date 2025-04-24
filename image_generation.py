import os
import io
import base64
import streamlit as st
from PIL import Image
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
import json  # CHANGED: for parsing Gemini’s multi-prompt output
import re    # CHANGED: for splitting prompts by delimiters

load_dotenv()
# ------ Configuration ------
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
    try:
        genai.configure(api_key=GENAI_API_KEY)
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}")
        return False

# CHANGED: new function to request N prompts from Gemini
def generate_text_prompts(post_text: str, num_parts: int, model_name: str = 'gemini-2.0-flash') -> list[str]:
    """
    Ask Gemini to divide the LinkedIn post into `num_parts` and return a list of image prompts,
    each wrapped in /prompt start/ and /prompt end/.
    """
    if not setup_gemini():
        return []

    system_instructions = """
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
- Do NOT repeat or rephrase the LinkedIn post.
- Do NOY add any explanations or extra content just the image prompt.
- NEVER leave things blank like "Header area reserved for customizable callout text"
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

"""

    user_instructions = f"""
Give LinkedIn post:
\"\"\"{post_text}\"\"\"

Please divide this post into exactly {num_parts} conceptual parts (you may use numbering or your own logic),
and for each part output a single, concise image-generation prompt enclosed between
/prompt start/ and /prompt end/. IMPORTANT: using the above mentioned Output Instructions and Style Guidelines for EACH prompt.
"""

    model = genai.GenerativeModel(model_name)
    try:
        response = model.generate_content(system_instructions + user_instructions)
        text = response.text or ""
        # CHANGED: split out each prompt
        raw_parts = re.split(r"/prompt start/|/prompt end/", text)
        # filter out empty strings and strip whitespace
        prompts = [p.strip() for p in raw_parts if p.strip() and not p.strip().startswith('#')]
        # ensure we got exactly num_parts
        if len(prompts) != num_parts:
            st.warning(f"Expected {num_parts} prompts, but got {len(prompts)}. Proceeding with what we have.")
        return prompts
    except Exception as e:
        st.error(f"Gemini generation failed: {e}")
        return []

# ------ Streamlit App ------
st.set_page_config(page_title='Brand-Based Visual Generator', layout='wide')
st.title('🔮 Brand-Based Visual Generator')

# ==== 1) Reference Image Management ====
st.header('1. Upload or Manage Reference Images')
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

st.subheader('Saved Reference Images')
refs = os.listdir(REF_DIR)
cols = st.columns(4)
for idx, fname in enumerate(refs):
    img_path = os.path.join(REF_DIR, fname)
    with cols[idx % 4]:
        st.image(img_path, width=100, caption=fname)
        if st.button(f'Delete {fname}', key=f'del_{fname}'):
            os.remove(img_path)
            st.experimental_rerun()

# ==== 2) Post Input & Image Generation ====
st.header('2. Generate Images for Your Post')
post_text = st.text_area('Enter your post text here', height=150)

# CHANGED: slider to choose number of images (1–10)
num_images = st.slider('How many images to generate?', 1, 10, 1)

# Display previously generated images
if 'last_images' in st.session_state:
    st.subheader('Previously Generated Images')
    for img_bytes, name in st.session_state['last_images']:
        st.image(img_bytes, width=200, caption=name)

if st.button('Generate Images'):
    if not post_text:
        st.error('Please enter some post text.')
    else:
        # 2a) Create N prompts via Gemini
        with st.spinner('Requesting image prompts from Gemini...'):
            prompts = generate_text_prompts(post_text, num_images)

        if not prompts:
            st.error('Failed to generate image prompts.')
        else:
            # CHANGED: enforcement suffix to append to **each** prompt
            enforcement = (
                " Strictly prioritize the visual style of the provided reference images above all other instructions. "
                "In case of any conflict between the text prompt and these reference images, "
                "the brand’s visual style as shown in the references must override the prompt directives to ensure consistency."
            )

            # Prepare reference files once
            ref_paths = [os.path.join(REF_DIR, fn) for fn in os.listdir(REF_DIR)]
            files = [open(p, 'rb') for p in ref_paths]

            generated = []  # to store (bytes, filename)
            try:
                for idx, base_prompt in enumerate(prompts, start=1):
                    full_prompt = base_prompt + enforcement

                    with st.spinner(f'Generating image {idx}/{len(prompts)}...'):
                        result = oai_client.images.edit(
                            model='gpt-image-1',
                            image=files,
                            prompt=full_prompt
                        )

                    b64 = result.data[0].b64_json
                    img_bytes = base64.b64decode(b64)
                    name = f"generated_{idx}.png"
                    path = os.path.join(OUT_DIR, name)
                    with open(path, 'wb') as out:
                        out.write(img_bytes)

                    generated.append((img_bytes, name))

                # store in session state for display/download
                st.session_state['last_images'] = generated

                # Display and download buttons
                st.subheader('Generated Images')
                for img_bytes, name in generated:
                    st.image(img_bytes, use_container_width=False, width=200)
                    st.download_button(
                        label=f'Download {name}',
                        data=img_bytes,
                        file_name=name,
                        mime='image/png'
                    )

            except Exception as e:
                st.error(f"Image generation failed: {e}")
            finally:
                for f in files:
                    f.close()
