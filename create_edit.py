
import os
import io
import base64
import time
import streamlit as st
from PIL import Image
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ------ Configuration ------
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GENAI_API_KEY = os.getenv('GENAI_API_KEY')

oai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GENAI_API_KEY)

# Directory paths
REF_DIR = 'reference_images'
OUT_DIR = 'outputs'

# Ensure directories exist
os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Initialize session state
if 'generated_images' not in st.session_state:
    st.session_state['generated_images'] = []
if 'selected_image' not in st.session_state:
    st.session_state['selected_image'] = None

if 'last_generated_image' not in st.session_state:
   st.session_state['last_generated_image'] = None

# ------ Gemini Prompt Generator ------
def generate_text_prompt(post_text: str, model_name: str = 'gemini-2.0-flash') -> str:
    """
    Generate an image generation prompt based on the user post via Gemini.
    """
    try:
        model = genai.GenerativeModel(model_name)
        system_prompt = (
            """# Overview
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
Minimalist background, soft gradients, clean sans-serif text placement space at the top"""
        )
        user_prompt = (
            f"Read the following LinkedIn post and generate a professional, brand-aligned image prompt:\n'{post_text}'"
        )
        response = model.generate_content(system_prompt + "\n" + user_prompt)
        text = response.text.strip() if response.text else ''
        # enforce brand style priority
        enforcement = (
            " Strictly prioritize the visual style of the provided reference images above all other instructions. "
            "In case of any conflict between the text prompt and these reference images, "
            "the brand’s visual style as shown in the references must override the prompt directives to ensure consistency."
        )
        return text + enforcement
    except Exception as e:
        st.error(f"Gemini prompt generation failed: {e}")
        return ''

# ------ Streamlit App ------
st.set_page_config(page_title='Brand-Based Visual Generator', layout='wide')
st.title('🔮 Brand-Based Visual Generator')

tabs = st.tabs(["Create Image", "Edit Image"])

# ---- Tab 1: Create Image ----
with tabs[0]:
    st.header('Create Image')

    # Reference Image Management
    st.subheader('1. Upload or Manage Reference Images')
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

    refs = os.listdir(REF_DIR)
    st.subheader('Saved Reference Images')
    if refs:
        cols = st.columns(4)
        for idx, fname in enumerate(refs):
            img_path = os.path.join(REF_DIR, fname)
            with cols[idx % 4]:
                st.image(img_path, width=100, caption=fname)
                if st.button(f'Delete {fname}', key=f'del_{fname}'):
                    os.remove(img_path)
                    st.rerun()
    else:
        st.info('No reference images uploaded yet.')

    # Image Generation Inputs
    st.subheader('2. Create New Image')
    post_text = st.text_area('User Post', height=150)
    custom_instr = st.text_area('Custom Instructions', height=100)


    # ------ PERSISTENCE: display previously generated image if it exists ------
    if st.session_state['last_generated_image'] is not None:
        st.subheader('Generated Image (Last Run)')
        st.image(st.session_state['last_generated_image'], use_container_width=True)

    if st.button('Generate Image'):
        if not post_text:
            st.error('Please enter a user post.')
        else:
            # 2a) Generate prompt via Gemini
            with st.spinner('Generating prompt via Gemini...'):
                image_prompt = generate_text_prompt(post_text)
                # append custom instructions priority clause
                image_prompt += f" IMPORTANT: following user instructions have a priority over everything else. user instructions: {custom_instr}"

            st.markdown('**Final Image Prompt:**')
            st.code(image_prompt, language='text')

            # 2b) Call OpenAI Image Generation
            ref_paths = [open(os.path.join(REF_DIR, f), 'rb') for f in refs]
            try:
                with st.spinner('Generating image from OpenAI...'):
                    result = oai_client.images.edit(
                        model='gpt-image-1',
                        prompt=image_prompt,
                        image=ref_paths
                    )
                b64 = result.data[0].b64_json
                img_bytes = base64.b64decode(b64)
                timestamp = int(time.time())
                name = f"generated_{timestamp}.png"
                out_path = os.path.join(OUT_DIR, name)
                with open(out_path, 'wb') as f:
                    f.write(img_bytes)



                # ------ PERSISTENCE: save into session state so it survives reruns ------
                st.session_state['last_generated_image'] = img_bytes
                # update session state
                st.session_state['generated_images'].append(name)
                st.session_state['selected_image'] = name

                # display
                st.subheader('Generated Image')
                st.image(img_bytes, use_container_width=True)


                out_name = f"generated_{len(os.listdir(OUT_DIR))+1}.png"
                    
                st.download_button(
                    label='Download Generated Image',
                    data=img_bytes,
                    file_name=out_name,
                    mime='image/png'
                )
                st.success(f'Saved to {out_path}')

            except Exception as e:
                st.error(f"Image generation failed: {e}")
            finally:
                for f in ref_paths:
                    f.close()

# ---- Tab 2: Edit Image ----
with tabs[1]:
    st.header('Edit Image')

     # ---- NEW: Upload any image to edit (in addition to generated ones) ----
    uploaded_edit = st.file_uploader(
        'Or upload your own image to edit (JPEG/PNG)',
        type=['jpg', 'jpeg', 'png'],
        help='This image will be added to the list below for editing.'
    )
    if uploaded_edit:
        edit_name = uploaded_edit.name
        edit_path = os.path.join(OUT_DIR, edit_name)
        # Save the upload if not already present
        if not os.path.exists(edit_path):
            with open(edit_path, 'wb') as f:
                f.write(uploaded_edit.getbuffer())
            st.success(f"Saved upload: {edit_name}")
            
        # Ensure it appears in our session-managed list (no rerun)
        if edit_name not in st.session_state['generated_images']:
            st.session_state['generated_images'].append(edit_name)
            st.session_state['selected_image'] = edit_name

    generated = st.session_state['generated_images']
    if generated:
        selection = st.selectbox(
            'Select an image to edit', generated,
            index=(generated.index(st.session_state['selected_image']) if st.session_state['selected_image'] in generated else 0)
        )
        st.session_state['selected_image'] = selection
        sel_path = os.path.join(OUT_DIR, selection)

        st.subheader('Selected Image')
        img = Image.open(sel_path)
        st.image(img, use_container_width=True)

        edit_instr = st.text_area('Edit Instructions', height=100)
        if st.button('Edit Image'):
            if not edit_instr:
                st.error('Please enter edit instructions.')
            else:
                try:
                    with st.spinner('Editing image...'):
                        with open(sel_path, 'rb') as img_file:
                            result = oai_client.images.edit(
                                model='gpt-image-1',
                                image=img_file,
                                prompt=edit_instr
                            )
                    b64 = result.data[0].b64_json
                    img_bytes = base64.b64decode(b64)
                    timestamp = int(time.time())
                    name = f"edited_{timestamp}.png"
                    out_path = os.path.join(OUT_DIR, name)
                    with open(out_path, 'wb') as f:
                        f.write(img_bytes)

                    st.session_state['generated_images'].append(name)
                    st.session_state['selected_image'] = name

                    st.subheader('Edited Image')
                    st.image(img_bytes, use_container_width=True)
                    out_name = f"generated_{len(os.listdir(OUT_DIR))+1}.png"

                    st.download_button(
                        label='Download Generated Image',
                        data=img_bytes,
                        file_name=out_name,
                        mime='image/png'
                    )


                    st.success(f'Saved edited image to {out_path}')
                except Exception as e:
                    st.error(f"Image edit failed: {e}")

        st.subheader('All Generated Images')
        cols = st.columns(min(4, len(generated)))
        for idx, fname in enumerate(generated):
            path = os.path.join(OUT_DIR, fname)
            with cols[idx % len(cols)]:
                st.image(path, width=100, caption=fname)
                if st.button(f'Select {fname}', key=f'sel_{fname}'):
                    st.session_state['selected_image'] = fname
                    st.rerun()
