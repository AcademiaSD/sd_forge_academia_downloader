import os
import json
import requests
import urllib.parse
import re
import math
import time
import tempfile
import threading
import gradio as gr
from modules import script_callbacks, shared, paths

# --- CONFIGURATION & PATHS ---
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
TOKENS_FILE = os.path.join(paths.data_path, "academia_tokens.json")

# Global dictionary for background threading
ACTIVE_DOWNLOADS = {}

def get_all_model_folders():
    core_folders =[
        "Stable-diffusion", "Lora", "VAE", "ControlNet", 
        "ControlNetPreprocessor", "diffusers", "embeddings", 
        "ESRGAN", "text_encoder"
    ]
    if os.path.exists(paths.models_path):
        try:
            for item in os.listdir(paths.models_path):
                if os.path.isdir(os.path.join(paths.models_path, item)):
                    if item not in core_folders:
                        core_folders.append(item)
        except: pass
    comfy_compatibility =["checkpoints", "loras", "vae", "controlnet", "text_encoders", "upscale_models", "unet", "diffusion_models"]
    for cf in comfy_compatibility:
        if cf not in core_folders:
            core_folders.append(cf)
    return core_folders

def get_target_path(folder_name, subfolder):
    base_paths = {
        "Stable-diffusion": getattr(shared.cmd_opts, 'ckpt_dir', None) or os.path.join(paths.models_path, "Stable-diffusion"),
        "checkpoints": getattr(shared.cmd_opts, 'ckpt_dir', None) or os.path.join(paths.models_path, "Stable-diffusion"),
        "Lora": getattr(shared.cmd_opts, 'lora_dir', None) or os.path.join(paths.models_path, "Lora"),
        "loras": getattr(shared.cmd_opts, 'lora_dir', None) or os.path.join(paths.models_path, "Lora"),
        "VAE": getattr(shared.cmd_opts, 'vae_dir', None) or os.path.join(paths.models_path, "VAE"),
        "vae": getattr(shared.cmd_opts, 'vae_dir', None) or os.path.join(paths.models_path, "VAE"),
        "ControlNet": getattr(shared.cmd_opts, 'controlnet_dir', None) or os.path.join(paths.models_path, "ControlNet"),
        "controlnet": getattr(shared.cmd_opts, 'controlnet_dir', None) or os.path.join(paths.models_path, "ControlNet"),
        "text_encoder": os.path.join(paths.models_path, "text_encoder"),
        "text_encoders": os.path.join(paths.models_path, "text_encoder"),
        "ESRGAN": os.path.join(paths.models_path, "ESRGAN"),
        "upscale_models": os.path.join(paths.models_path, "ESRGAN"),
        "embeddings": getattr(shared.cmd_opts, 'embeddings_dir', None) or os.path.join(paths.data_path, "embeddings")
    }
    target_base = base_paths.get(folder_name, os.path.join(paths.models_path, folder_name))
    if subfolder:
        safe_subfolder = subfolder.replace("..", "").strip("\\/")
        target_base = os.path.join(target_base, safe_subfolder)
    return target_base

def format_size(size_bytes):
    try:
        size_bytes = int(size_bytes)
        if size_bytes == 0: return "0 B"
        size_name = ("B", "KB", "MB", "GB", "TB")
        i = int(math.floor(math.log(size_bytes, 1024))) if size_bytes > 0 else 0
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"
    except:
        return "Unknown"

def get_headers_with_auth(url, civitai_token="", hf_token=""):
    req_headers = HEADERS.copy()
    if "civitai.com" in url and civitai_token:
        req_headers["Authorization"] = f"Bearer {civitai_token}"
    elif "huggingface.co" in url and hf_token:
        req_headers["Authorization"] = f"Bearer {hf_token}"
    return req_headers

def get_file_info_from_url(url, civitai_token="", hf_token=""):
    if not url or not url.startswith(('http://', 'https://')):
        return "Invalid Link", "0 B"
    try:
        req_headers = get_headers_with_auth(url, civitai_token, hf_token)
        response = requests.get(url, stream=True, allow_redirects=True, headers=req_headers, timeout=8)
        response.close()
        
        if response.status_code in[401, 403] or "civitai.com/login" in response.url:
            return None, "Auth Required"
            
        size_bytes = response.headers.get('Content-Length')
        formatted_size = format_size(size_bytes) if size_bytes else "Unknown"

        fname = None
        cd = response.headers.get('Content-Disposition')
        if cd:
            match = re.findall('filename="?([^"]+)"?', cd)
            if match: fname = match[0]
            
        if not fname:
            parsed = urllib.parse.urlparse(response.url)
            fname = os.path.basename(parsed.path)
            if not fname or fname.isdigit():
                fname = (fname if fname else "model") + ".safetensors"
        return fname, formatted_size
    except:
        return "unknown_error.safetensors", "Unknown"

def state_to_df(state_list):
    df =[]
    for item in state_list:
        df.append([
            item.get("filename", "Pending..."),
            item.get("folder", ""),
            item.get("subfolder", ""),
            item.get("filesize", "-- MB"),
            item.get("status", "Not checked")
        ])
    return df

def update_ui_state(state):
    df = state_to_df(state)
    choices =[]
    for i, item in enumerate(state):
        choices.append(f"{i} - {item.get('filename', 'Pending...')}")
    dropdown_val = choices[0] if choices else None
    return state, df, gr.update(choices=choices, value=dropdown_val)

# --- BACKGROUND DOWNLOAD ENGINE ---
def background_download(real_url, file_path, temp_path, civ_token, hf_token):
    try:
        req_headers = get_headers_with_auth(real_url, civ_token, hf_token)
        with requests.get(real_url, stream=True, allow_redirects=True, headers=req_headers) as r:
            r.raise_for_status()
            total_length = r.headers.get('content-length')
            total_length = int(total_length) if total_length else 0
            downloaded = 0
            
            with open(temp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_length > 0:
                            pct = int((downloaded / total_length) * 100)
                            ACTIVE_DOWNLOADS[real_url] = f"⏳ {pct}%"
                        else:
                            mb = downloaded // (1024 * 1024)
                            ACTIVE_DOWNLOADS[real_url] = f"⏳ {mb} MB"
                            
        os.replace(temp_path, file_path)
        ACTIVE_DOWNLOADS[real_url] = "✅ Ready"
    except Exception as e:
        print(f"[AcademiaSD] Error downloading: {e}")
        ACTIVE_DOWNLOADS[real_url] = "❌ Error"
        if os.path.exists(temp_path):
            os.remove(temp_path)

# --- GRADIO UI ---
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as academia_tab:
        
        model_state = gr.State([]) 
        
        gr.Markdown("## ⬇️ Academia SD Automatic Downloader")
        gr.Markdown("Automatically download and install models. Load a `.json` preset or create your own list.")
        
        # 1. TOKENS ROW
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🔑 API Keys (Optional)")
                    with gr.Row():
                        civitai_input = gr.Textbox(label="Civitai API Key", type="password", placeholder="Paste token here...", scale=2)
                        hf_input = gr.Textbox(label="HuggingFace Token", type="password", placeholder="Paste token here...", scale=2)
                        save_tokens_btn = gr.Button("💾 Save Tokens", scale=1)
            
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📂 Presets (.json)")
                    import_json = gr.File(label="Upload Preset", file_types=[".json"])
                    export_btn = gr.Button("💾 Export Current List", variant="secondary")
                    export_file = gr.File(label="Download Preset", visible=False)

        gr.Markdown("---")
        
        # 2. ADD MODEL SECTION
        with gr.Group():
            gr.Markdown("### ➕ Add Model to List")
            with gr.Row():
                url_input = gr.Textbox(label="Model URL (Civitai or HF Repo)", placeholder="https://civitai.com/... or HuggingFace", scale=3)
                hf_file_dropdown = gr.Dropdown(label="Select File (HuggingFace)", choices=[], visible=False, scale=2)
            
            with gr.Row():
                folder_input = gr.Dropdown(choices=get_all_model_folders(), label="Destination Folder", value="Stable-diffusion", scale=1)
                subfolder_input = gr.Textbox(label="Subfolder (Optional)", placeholder="e.g. Anime", scale=1)
            
            # MAGIA VISUAL: Botón Add to list centrado y compacto usando columnas fantasma
            with gr.Row():
                gr.Column(scale=2) # Espaciador izquierdo
                add_btn = gr.Button("➕ Add to List", variant="primary", scale=1)
                gr.Column(scale=2) # Espaciador derecho

        gr.Markdown("---")
        gr.Markdown("### 📋 Download Queue")
        
        # 3. TABLE
        model_df = gr.Dataframe(
            headers=["File Name", "Folder", "Subfolder", "Size", "Status"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True
        )
        
        # 4. CONTROLS
        with gr.Row():
            refresh_btn = gr.Button("🔄 Check Status", elem_id="asd_refresh_btn")
            clear_btn = gr.Button("🗑️ Clear Entire List")
            
        with gr.Group():
            with gr.Row():
                remove_dropdown = gr.Dropdown(label="Remove Model", show_label=False, choices=[], scale=3)
                remove_btn = gr.Button("❌ Remove Selected", scale=1)

        download_btn = gr.Button("⬇️ DOWNLOAD ALL MISSING", variant="primary", size="lg")
        
        # Hidden component to stop JS polling
        stop_signal = gr.Textbox(value="stop", visible=False, elem_id="asd_stop_signal")

        stop_signal.change(fn=None, inputs=[stop_signal], outputs=None, _js="""
            function(val) {
                if(val === 'stop' && window.asd_interval) {
                    clearInterval(window.asd_interval);
                    window.asd_interval = null;
                }
                return[];
            }
        """)

        # --- LOGIC & EVENTS ---
        def load_tokens():
            if os.path.exists(TOKENS_FILE):
                try:
                    with open(TOKENS_FILE, "r") as f:
                        data = json.load(f)
                        return data.get("civitai", ""), data.get("huggingface", "")
                except: pass
            return "", ""
        
        academia_tab.load(fn=load_tokens, outputs=[civitai_input, hf_input])

        def save_tokens(civ, hf):
            with open(TOKENS_FILE, "w") as f:
                json.dump({"civitai": civ, "huggingface": hf}, f)
            return gr.update(value="✅ Saved!")
            
        save_tokens_btn.click(fn=save_tokens, inputs=[civitai_input, hf_input], outputs=[save_tokens_btn])

        def parse_url_change(url, hf_token):
            if "huggingface.co" in url and "/resolve/" not in url and "/blob/" not in url:
                match = re.search(r"huggingface\.co/([^/]+/[^/?#]+)(?:/tree/([^/?#]+))?", url)
                if match:
                    repo_id = match.group(1)
                    branch = match.group(2) if match.group(2) else "main"
                    api_url = f"https://huggingface.co/api/models/{repo_id}"
                    headers = HEADERS.copy()
                    if hf_token: headers["Authorization"] = f"Bearer {hf_token}"
                    try:
                        res = requests.get(api_url, headers=headers, timeout=10)
                        if res.status_code == 200:
                            data = res.json()
                            siblings = data.get("siblings",[])
                            valid_exts = (".safetensors", ".gguf", ".ckpt", ".pt", ".bin", ".pth", ".onnx", ".sft")
                            files =[]
                            for s in siblings:
                                fname = s["rfilename"]
                                if fname.endswith(valid_exts):
                                    files.append((fname, f"https://huggingface.co/{repo_id}/resolve/{branch}/{fname}"))
                            if files:
                                choices = [f[0] for f in files]
                                return gr.update(choices=choices, value=choices[0], visible=True)
                    except: pass
            return gr.update(choices=[], visible=False)

        url_input.change(fn=parse_url_change, inputs=[url_input, hf_input], outputs=[hf_file_dropdown])

        def refresh_status(state, civ_token, hf_token):
            any_downloading = False
            for item in state:
                real_url = item.get("selected_url")
                if not real_url: continue
                
                if real_url in ACTIVE_DOWNLOADS:
                    item["status"] = ACTIVE_DOWNLOADS[real_url]
                    if "⏳" in item["status"]:
                        any_downloading = True
                    elif item["status"] == "✅ Ready":
                        target_dir = get_target_path(item["folder"], item["subfolder"])
                        full_path = os.path.join(target_dir, item["filename"])
                        if os.path.exists(full_path):
                            try: item["filesize"] = format_size(os.path.getsize(full_path))
                            except: pass
                    continue
                    
                if item.get("status") not in["✅ Ready"]:
                    fname, fsize = get_file_info_from_url(real_url, civ_token, hf_token)
                    if not fname:
                        item["status"] = "❌ Need API Key"
                    else:
                        item["filename"] = fname
                        item["filesize"] = fsize
                        target_dir = get_target_path(item["folder"], item["subfolder"])
                        full_path = os.path.join(target_dir, fname)
                        
                        if os.path.exists(full_path):
                            try: item["filesize"] = format_size(os.path.getsize(full_path))
                            except: pass
                            item["status"] = "✅ Ready"
                        elif "Error" in item.get("status", ""):
                            pass
                        else:
                            item["status"] = "🔴 Missing"
            
            signal = "run" if any_downloading else "stop"
            st, df, drop = update_ui_state(state)
            return st, df, drop, signal

        def add_model(state, url, hf_file, folder, subfolder, civ_token, hf_token):
            if not url: return update_ui_state(state) + ("stop",)
            
            real_url = url
            filename = "Pending..."
            if hf_file:
                match = re.search(r"huggingface\.co/([^/]+/[^/?#]+)(?:/tree/([^/?#]+))?", url)
                if match:
                    repo_id = match.group(1)
                    branch = match.group(2) if match.group(2) else "main"
                    real_url = f"https://huggingface.co/{repo_id}/resolve/{branch}/{hf_file}"
                    filename = hf_file

            state.append({
                "url": url, 
                "selected_url": real_url, 
                "filename": filename,
                "filesize": "-- MB",
                "folder": folder,
                "subfolder": subfolder,
                "status": "Checking..."
            })
            return refresh_status(state, civ_token, hf_token)

        add_btn.click(fn=add_model, inputs=[model_state, url_input, hf_file_dropdown, folder_input, subfolder_input, civitai_input, hf_input], outputs=[model_state, model_df, remove_dropdown, stop_signal])

        def clear_state(): return update_ui_state([]) + ("stop",)
        clear_btn.click(fn=clear_state, outputs=[model_state, model_df, remove_dropdown, stop_signal])
        
        def remove_individual_model(state, selection):
            if not selection or not state: return update_ui_state(state) + ("stop",)
            try:
                idx = int(selection.split(" - ")[0])
                state.pop(idx)
            except: pass
            return update_ui_state(state) + ("stop",)
            
        remove_btn.click(fn=remove_individual_model, inputs=[model_state, remove_dropdown], outputs=[model_state, model_df, remove_dropdown, stop_signal])

        # --- EXPORTAR CON NOMBRE DINÁMICO ---
        def export_json_file(state):
            if not state: return gr.update(visible=False)
            
            # Nombre dinámico basado en el primer elemento
            first_item_name = state[0].get("filename", "models")
            
            if first_item_name and first_item_name != "Pending...":
                # Quitar extensión .safetensors, .gguf, etc.
                clean_name = re.sub(r'\.[^.]+$', '', first_item_name)
            else:
                # Fallback: intentar sacarlo de la URL
                url = state[0].get("url", "")
                parsed = urllib.parse.urlparse(url)
                base = os.path.basename(parsed.path)
                if base:
                    clean_name = re.sub(r'\.[^.]+$', '', base)
                else:
                    clean_name = "models"
                    
            # Eliminar caracteres inválidos para Windows/Mac
            clean_name = re.sub(r'[\\/*?:"<>|]', "", clean_name)
            if not clean_name: clean_name = "models"
                
            file_name = f"download_list_{clean_name}.json"
            tmp_path = os.path.join(tempfile.gettempdir(), file_name)
            
            export_data =[{"url": s["url"], "selected_url": s["selected_url"], "filename": s["filename"], "filesize": s["filesize"], "folder": s["folder"], "subfolder": s["subfolder"]} for s in state]
            with open(tmp_path, "w") as f:
                json.dump(export_data, f, indent=4)
                
            return gr.update(value=tmp_path, visible=True)

        export_btn.click(fn=export_json_file, inputs=[model_state], outputs=[export_file])

        def import_json_file(file_obj, current_state, civ, hf):
            if file_obj is None: return update_ui_state(current_state) + ("stop",)
            try:
                with open(file_obj.name, "r") as f:
                    data = json.load(f)
                new_state =[]
                for item in data:
                    new_state.append({
                        "url": item.get("url", ""),
                        "selected_url": item.get("selected_url", item.get("url", "")),
                        "filename": item.get("filename", "Pending..."),
                        "filesize": item.get("filesize", "-- MB"),
                        "folder": item.get("folder", "Stable-diffusion"),
                        "subfolder": item.get("subfolder", ""),
                        "status": "Not checked"
                    })
                return refresh_status(new_state, civ, hf)
            except:
                return update_ui_state(current_state) + ("stop",)

        import_json.upload(fn=import_json_file, inputs=[import_json, model_state, civitai_input, hf_input], outputs=[model_state, model_df, remove_dropdown, stop_signal])

        refresh_btn.click(fn=refresh_status, inputs=[model_state, civitai_input, hf_input], outputs=[model_state, model_df, remove_dropdown, stop_signal])

        def start_downloads(state, civ_token, hf_token):
            state, _, _, _ = refresh_status(state, civ_token, hf_token)

            for item in state:
                if item["status"] == "✅ Ready" or "Need API Key" in item["status"] or "Error" in item["status"]:
                    continue 
                
                real_url = item["selected_url"]
                fname = item["filename"]
                target_dir = get_target_path(item["folder"], item["subfolder"])
                os.makedirs(target_dir, exist_ok=True)
                
                file_path = os.path.join(target_dir, fname)
                temp_path = file_path + ".temp"

                item["status"] = "⏳ Starting..."
                ACTIVE_DOWNLOADS[real_url] = "⏳ Starting..."
                
                t = threading.Thread(target=background_download, args=(real_url, file_path, temp_path, civ_token, hf_token))
                t.start()
                
            st, df, drop = update_ui_state(state)
            return st, df, drop, "run"

        download_btn.click(
            fn=start_downloads, 
            inputs=[model_state, civitai_input, hf_input], 
            outputs=[model_state, model_df, remove_dropdown, stop_signal],
            _js="""
            function(a, b, c) {
                if (window.asd_interval) clearInterval(window.asd_interval);
                window.asd_interval = setInterval(() => {
                    const refresh = document.querySelector('#asd_refresh_btn');
                    if (refresh) refresh.click();
                }, 2000);
                return [a, b, c];
            }
            """
        )

    return[(academia_tab, "Academia Downloader ⬇️", "academia_downloader_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)