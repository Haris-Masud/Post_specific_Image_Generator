import base64
import os
import io
import tempfile
import time
import streamlit as st
from PIL import Image
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
from pymongo import MongoClient
import gridfs
import hashlib
import uuid


st.set_page_config(page_title="Brand Based Social Media Image Generator", layout="wide")

# ─── Load environment ─────────────────────────────────────────────────────────
load_dotenv()
MONGO_URI      = os.getenv("MONGODB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GENAI_API_KEY  = os.getenv("GENAI_API_KEY")


# Directory paths (near where you create OUT_DIR)
REF_DIR = "reference_images"
os.makedirs(REF_DIR, exist_ok=True)
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Init OpenAI & Gemini clients ────────────────────────────────────────────
oai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GENAI_API_KEY)

# ─── Init MongoDB + GridFS ───────────────────────────────────────────────────
mongo = MongoClient(MONGO_URI)
db    = mongo["Image_generation"]
chats = db["chats"]     # stores: {_id, name}
ref_fs = gridfs.GridFS(db, collection="refs")
gen_fs = gridfs.GridFS(db, collection="generated")

try:
    ref_fs._GridFS__files.create_index(
        [("filename",1), ("metadata.chat",1)],
        unique=True
    )
except Exception:
    pass

# ─── Session-state defaults ──────────────────────────────────────────────────
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "creating_chat" not in st.session_state:
    st.session_state.creating_chat = False
if "renaming_chat" not in st.session_state:
    st.session_state.renaming_chat = False
if "deleting_chat" not in st.session_state:
    st.session_state.deleting_chat = False

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

if "processed_upload_hashes" not in st.session_state:
    st.session_state.processed_upload_hashes = set()

if "last_generated" not in st.session_state:
    st.session_state.last_generated = None



if "last_chat" not in st.session_state:
    st.session_state.last_chat = None

if "uploader_key" not in st.session_state:
    # start with a random key
    st.session_state.uploader_key = f"uploader_{uuid.uuid4()}"


if "edit_uploader_key" not in st.session_state:
    st.session_state.edit_uploader_key = f"edit_uploader_{uuid.uuid4()}"


# ─── Sidebar: Chat (Project) Management ───────────────────────────────────────
with st.sidebar:
    st.header("📂 Projects")
    all_chats = list(chats.find({}, {"_id":0, "name":1}))
    names     = [c["name"] for c in all_chats]
    
    # Select existing
    if names:
        sel = st.selectbox("Open project", names, index=names.index(st.session_state.current_chat) if st.session_state.current_chat in names else 0)
        st.session_state.current_chat = sel
    else:
        st.info("No projects yet")

    st.markdown("---")
    # Create
    if st.button("➕ New Project"):
        st.session_state.creating_chat = True
        st.session_state.renaming_chat = False
        st.session_state.deleting_chat = False

    if st.session_state.creating_chat:
        new_name = st.text_input("Project name", key="new_name")
        if st.button("Create"):
            if new_name.strip():
                chats.insert_one({"name": new_name.strip()})
                st.session_state.current_chat = new_name.strip()
                st.session_state.creating_chat = False
                st.rerun()

    # Rename
    # if st.button("✏️ Rename Project"):
    #     st.session_state.renaming_chat = True
    #     st.session_state.creating_chat = False
    #     st.session_state.deleting_chat = False

    # if st.session_state.renaming_chat and st.session_state.current_chat:
    #     new_name = st.text_input("New name for project", key="rename_name")
    #     if st.button("Rename", key="do_rename"):
    #         if new_name.strip():
    #             chats.update_one({"name": st.session_state.current_chat}, {"$set": {"name": new_name.strip()}})
    #             st.session_state.current_chat = new_name.strip()
    #             st.session_state.renaming_chat = False
    #             st.rerun()

    # Delete
    if st.button("🗑️ Delete Project"):
        st.session_state.deleting_chat = True
        st.session_state.creating_chat = False
        st.session_state.renaming_chat = False

    if st.session_state.deleting_chat and st.session_state.current_chat:
        confirm = st.checkbox(f"Confirm delete '{st.session_state.current_chat}'")
        if confirm and st.button("Delete forever"):
            name = st.session_state.current_chat
            chats.delete_one({"name": name})
            # remove associated images
            for f in ref_fs.find({"metadata.chat": name}):
                ref_fs.delete(f._id)
            for f in gen_fs.find({"metadata.chat": name}):
                gen_fs.delete(f._id)
            st.session_state.current_chat = None
            st.session_state.deleting_chat = False
            st.rerun()


    if st.session_state.current_chat != st.session_state.last_chat:
        # project changed → reset uploader
        st.session_state.uploader_key = f"uploader_{uuid.uuid4()}"
        st.session_state.edit_uploader_key = f"edit_uploader_{uuid.uuid4()}"
        st.session_state.last_chat = st.session_state.current_chat

# ─── Helpers to fetch images from GridFS ───────────────────────────────────────
def list_ref_images(chat_name):
    return list(ref_fs.find({"metadata.chat": chat_name}))

def list_generated_images(chat_name):
    return list(gen_fs.find({"metadata.chat": chat_name}))

# ─── Gemini Prompt Generator ─────────────────────────────────────────────────
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
- NEVER leave things blank or any placeholders like "Header area reserved for customizable callout text"
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

# ─── Main App ─────────────────────────────────────────────────────────────────

st.title("🔮 Brand Based Social Media Image Generator")

if not st.session_state.current_chat:
    st.warning("Please create or select a project in the sidebar first.")
    st.stop()

# Tabs for Create / Edit
tabs = st.tabs(["Create Image", "Edit Image"])

# ─── Tab 1: Create Image ──────────────────────────────────────────────────────
with tabs[0]:
    st.header(f"Create Image — {st.session_state.current_chat}")

    # 1) Upload/manage references
    st.subheader("1. Upload / Delete Reference Images")
    # uploaded = st.file_uploader("Select JPG/PNG", type=["jpg","jpeg","png"], accept_multiple_files=True)
    # if uploaded:
    #     for up in uploaded:
    #         ref_fs.put(up.read(),
    #                    filename=up.name,
    #                    metadata={"chat": st.session_state.current_chat})
    #     st.success(f"Stored {len(uploaded)} images for '{st.session_state.current_chat}'")
    uploaded = st.file_uploader("Select JPG/PNG", type=["jpg","jpeg","png"], accept_multiple_files=True, key=st.session_state.uploader_key)
    if uploaded:
        new_count = 0
        for up in uploaded:
            data = up.read()  # bytes
            digest = hashlib.sha256(data).hexdigest()

            # Client-side: skip if we've processed this content already
            if digest in st.session_state.processed_upload_hashes:
                continue

            # Server-side: skip if identical hash exists in GridFS
            exists = ref_fs.find_one({
                "metadata.chat": st.session_state.current_chat,
                "metadata.hash": digest
            })
            if not exists:
                ref_fs.put(
                    data,
                    filename=up.name,
                    metadata={"chat": st.session_state.current_chat, "hash": digest}
                )
                new_count += 1

            # Mark this digest so we never re-check in this session
            st.session_state.processed_upload_hashes.add(digest)

        if new_count:
            st.success(f"Stored {new_count} new images for '{st.session_state.current_chat}'")
        else:
            st.info("No new images to store.")
#*********************************************************************************************************

    # show existing refs
    refs = list_ref_images(st.session_state.current_chat)
    if refs:
        cols = st.columns(4)
        for idx, gf in enumerate(refs):
            data = gf.read()
            with cols[idx % 4]:
                st.image(data, width=100, caption=gf.filename)
                if st.button(f"Delete", key=f"delref_{gf._id}"):
                    ref_fs.delete(gf._id)
                    st.rerun()
    else:
        st.info("No reference images yet.")

    # 2) Create new image
    st.subheader("2. Generate New Image")
    post_txt     = st.text_area("User Post", height=150)
    custom_instr = st.text_area("User Prompt", height=100)

    if st.button("Generate Image"):
        if not post_txt.strip():
            st.error("Enter a post first.")
        else:
            with st.spinner("Generating prompt..."):
                # prompt = generate_text_prompt(post_txt)
                prompt = f"You are a graphic designer creating a marketing image for a LinkedIn post. Create an image based on the following instructions (Follow them closely and carefully as they have the !highest! priority):\n {custom_instr} \n. For reference, you are generating an accompanying image for following LinkedIn post: {post_txt}"
                enforcement = (
                """ Strictly prioritize the visual style of the provided reference images.
                The brand's visual style should be followed as shown in the references to ensure consistency."""
                )
                prompt += enforcement

            # — re-fetch fresh GridOuts to get unread data —
            fresh_refs = list_ref_images(st.session_state.current_chat)

            if not fresh_refs:
                st.error("No reference images available.")
                st.stop()

            # ▶ Dump each fresh GridFS ref image back to disk
            temp_paths = []
            for gf in fresh_refs:
                path = os.path.join(REF_DIR, gf.filename)
                with open(path, "wb") as f:
                    f.write(gf.read())
                temp_paths.append(path)

            # Open real files for OpenAI
            file_handles = [open(p, "rb") for p in temp_paths]

            try:
                with st.spinner('Generating image from OpenAI...'):
                    result = oai_client.images.edit(
                        model='gpt-image-1',
                        prompt=prompt,
                        image=file_handles,
                    )
                img_b64  = result.data[0].b64_json
                img_bytes= base64.b64decode(img_b64)

                # persist to GridFS
                gen_fs.put(img_bytes,
                           filename=f"gen_{int(time.time())}.png",
                           metadata={"chat": st.session_state.current_chat})

                st.session_state.last_generated = img_bytes
                st.success("Image generated and stored.")

                # display
                st.subheader("Generated Image")
                st.image(img_bytes, use_container_width=True)

                random_name = os.urandom(4).hex()
                out_name = f"generated_img_{random_name}.png"
                    
                st.download_button(
                    label='Download Generated Image',
                    data=img_bytes,
                    file_name=out_name,
                    mime='image/png'
                )
                st.success(f'Image saved as {out_name}')

            except Exception as e:
                st.error("OpenAI image.edit failed: " + str(e))
            finally:
                # close & delete the on-disk files
                for fh in file_handles:
                    fh.close()
                for p in temp_paths:
                    os.remove(p)

# ─── Tab 2: Edit Image ────────────────────────────────────────────────────────
with tabs[1]:
    st.header(f"Edit Image — {st.session_state.current_chat}")

    # allow uploading an external image to edit
    up = st.file_uploader("Or upload an image to edit", type=["jpg","jpeg","png"], key=st.session_state.edit_uploader_key)
    if up:
        bin = up.read()
        gen_fs.put(bin,
                   filename=up.name,
                   metadata={"chat": st.session_state.current_chat})
        st.success(f"Added '{up.name}' to editable images.")

    # list all generated for this chat
    gens = list_generated_images(st.session_state.current_chat)
    if gens:
        # selection
        names = [g.filename for g in gens]
        sel   = st.selectbox("Select image to edit", names,
                             index=names.index(st.session_state.selected_image)
                                   if st.session_state.selected_image in names else 0)
        st.session_state.selected_image = sel

        # show large
        sel_gf = next(g for g in gens if g.filename == sel)
        img = Image.open(io.BytesIO(sel_gf.read()))
        st.image(img, use_container_width=True)

        # edit instructions
        instr = st.text_area("Edit Instructions", height=100)
        if st.button("Edit Image"):
            if not instr:
                st.error("Enter instructions")
            else:
                # List generated images for the current chat
                gens = list_generated_images(st.session_state.current_chat)
                
                if not gens:
                    st.error("No images available to edit.")
                    st.stop()

                # get selected image name from session state
                selected_image_name = st.session_state.selected_image

                # Find the selected image from GridFS
                sel_gf = next(g for g in gens if g.filename == selected_image_name)
                img_data = sel_gf.read()

                # Save the selected image to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    temp_file.write(img_data)
                    temp_file_path = temp_file.name

                # Open the temporary file in 'rb' mode to pass to OpenAI
                with open(temp_file_path, "rb") as f:
                    with st.spinner("Calling OpenAI Edit..."):
                        result = oai_client.images.edit(
                            model="gpt-image-1",
                            image=f,
                            prompt=instr
                        )
                new_bytes = base64.b64decode(result.data[0].b64_json)
                filename=f"edit_{int(time.time())}.png"
                gen_fs.put(new_bytes,
                           filename=filename,
                           metadata={"chat": st.session_state.current_chat})
                st.success("Edited image stored.")


                st.session_state.selected_image = filename

                random_name = os.urandom(4).hex()
                out_name = f"generated_img_{random_name}.png"
                    

                st.success(f'Image saved as {out_name}')
                st.rerun()


        # Download button to download selected image
        if st.button("Download Selected Image"):
            if st.session_state.selected_image:
                # Fetch selected image from GridFS
                gens = list_generated_images(st.session_state.current_chat)
                selected_image = next(g for g in gens if g.filename == st.session_state.selected_image)
                img_data = selected_image.read()

                # Provide download button for the selected image
                st.download_button(
                    label="Download Image",
                    data=img_data,
                    file_name=selected_image.filename,
                    mime="image/png"
                )
            else:
                st.warning("Please select an image to download.")


        # Thumbnails
        st.subheader("All Images")
        gens = list_generated_images(st.session_state.current_chat)
        cols = st.columns(min(4, len(gens)))
        for idx, gf in enumerate(gens):
            thumb = Image.open(io.BytesIO(gf.read()))
            with cols[idx % 4]:
                st.image(thumb, width=100, caption=gf.filename)
                
                # Select Button
                if st.button(f"Select", key=f"sel_{gf._id}"):
                    st.session_state.selected_image = gf.filename
                    st.rerun()

                # Delete Button
                if st.button(f"Delete", key=f"del_{gf._id}"):
                    # Delete image from GridFS
                    gen_fs.delete(gf._id)
                    st.success(f"Deleted {gf.filename}")
                    st.rerun()  # Refresh the UI after deletion

    else:
        st.info("No images to edit—generate or upload one first.")
