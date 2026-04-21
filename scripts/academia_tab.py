import os
import json
import requests
import urllib.parse
import re
import math
import tempfile
import gradio as gr
from modules import script_callbacks, shared, paths

# --- CONFIGURACIÓN Y RUTAS ---
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
TOKENS_FILE = os.path.join(paths.data_path, "academia_tokens.json")

def get_all_model_folders():
    """
    Escanea la carpeta models/ de Forge y fuerza las carpetas estándar
    para que siempre aparezcan en el desplegable.
    """
    # 1. Carpetas oficiales de Forge/A1111 (basadas en tu captura)
    core_folders =[
        "Stable-diffusion", "Lora", "VAE", "ControlNet", 
        "ControlNetPreprocessor", "diffusers", "embeddings", 
        "ESRGAN", "text_encoder"
    ]
    
    # 2. Escanear el disco duro para buscar subcarpetas nuevas que haya creado el usuario
    if os.path.exists(paths.models_path):
        try:
            for item in os.listdir(paths.models_path):
                if os.path.isdir(os.path.join(paths.models_path, item)):
                    if item not in core_folders:
                        core_folders.append(item)
        except:
            pass
            
    # 3. Añadir las carpetas de ComfyUI al final para garantizar la compatibilidad cruzada de los .json
    comfy_compatibility =["checkpoints", "loras", "vae", "controlnet", "text_encoders", "upscale_models", "unet", "diffusion_models"]
    for cf in comfy_compatibility:
        if cf not in core_folders:
            core_folders.append(cf)
            
    return core_folders

def get_target_path(folder_name, subfolder):
    """
    Traductor Universal: Mapea las carpetas elegidas a sus rutas reales en el disco duro.
    Si lee "checkpoints" (de Comfy), lo guardará en "Stable-diffusion" (de Forge).
    """
    base_paths = {
        # Modelos Principales
        "Stable-diffusion": getattr(shared.cmd_opts, 'ckpt_dir', None) or os.path.join(paths.models_path, "Stable-diffusion"),
        "checkpoints": getattr(shared.cmd_opts, 'ckpt_dir', None) or os.path.join(paths.models_path, "Stable-diffusion"),
        
        # LoRAs
        "Lora": getattr(shared.cmd_opts, 'lora_dir', None) or os.path.join(paths.models_path, "Lora"),
        "loras": getattr(shared.cmd_opts, 'lora_dir', None) or os.path.join(paths.models_path, "Lora"),
        
        # VAEs
        "VAE": getattr(shared.cmd_opts, 'vae_dir', None) or os.path.join(paths.models_path, "VAE"),
        "vae": getattr(shared.cmd_opts, 'vae_dir', None) or os.path.join(paths.models_path, "VAE"),
        
        # ControlNet
        "ControlNet": getattr(shared.cmd_opts, 'controlnet_dir', None) or os.path.join(paths.models_path, "ControlNet"),
        "controlnet": getattr(shared.cmd_opts, 'controlnet_dir', None) or os.path.join(paths.models_path, "ControlNet"),
        
        # Text Encoders
        "text_encoder": os.path.join(paths.models_path, "text_encoder"),
        "text_encoders": os.path.join(paths.models_path, "text_encoder"),
        
        # Upscalers
        "ESRGAN": os.path.join(paths.models_path, "ESRGAN"),
        "upscale_models": os.path.join(paths.models_path, "ESRGAN"),
        
        # Embeddings
        "embeddings": getattr(shared.cmd_opts, 'embeddings_dir', None) or os.path.join(paths.data_path, "embeddings")
    }
    
    # Si la carpeta no está en el diccionario, la crea en la raíz de models/
    target_base = base_paths.get(folder_name, os.path.join(paths.models_path, folder_name))
    
    # Añade la subcarpeta si el usuario escribió una
    if subfolder:
        safe_subfolder = subfolder.replace("..", "").strip("\\/")
        target_base = os.path.join(target_base, safe_subfolder)
        
    return target_base

# --- UTILIDADES DE DESCARGA E INFORMACIÓN ---
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
        
        if response.status_code in [401, 403] or "civitai.com/login" in response.url:
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
    except Exception as e:
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

# --- INTERFAZ GRÁFICA (GRADIO) ---
def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as academia_tab:
        
        model_state = gr.State([]) 
        
        gr.Markdown("## ⬇️ Academia SD Automatic Downloader")
        gr.Markdown("Descarga e instala modelos automáticamente. Carga un *Preset* `.json` de clase o crea tu propia lista.")
        
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### 🔑 API Keys (Optional)")
                    with gr.Row():
                        civitai_input = gr.Textbox(label="Civitai API Key", type="password", placeholder="Paste token here...")
                        hf_input = gr.Textbox(label="HuggingFace Token", type="password", placeholder="Paste token here...")
                    save_tokens_btn = gr.Button("💾 Save Tokens", variant="secondary")
            
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📂 Presets (.json)")
                    import_json = gr.File(label="Upload Preset", file_types=[".json"])
                    export_btn = gr.Button("💾 Export Current List")
                    export_file = gr.File(label="Download Preset", visible=False)

        with gr.Group():
            gr.Markdown("### ➕ Add Model to List")
            with gr.Row():
                url_input = gr.Textbox(label="Model URL", placeholder="https://civitai.com/... or HF Repo", scale=3)
                hf_file_dropdown = gr.Dropdown(label="Select File (HuggingFace)", choices=[], visible=False, scale=2)
            
            with gr.Row():
                # Usamos la función dinámica para llenar el Dropdown
                folder_input = gr.Dropdown(
                    choices=get_all_model_folders(), 
                    label="Destination Folder", value="Stable-diffusion", scale=2
                )
                subfolder_input = gr.Textbox(label="Subfolder (Optional)", placeholder="e.g. Anime", scale=2)
                add_btn = gr.Button("➕ Add to List", variant="primary", scale=1)

        gr.Markdown("### 📋 Download Queue")
        model_df = gr.Dataframe(
            headers=["File Name", "Folder", "Subfolder", "Size", "Status"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True
        )
        
        with gr.Row():
            refresh_btn = gr.Button("🔄 Check Status")
            clear_btn = gr.Button("🗑️ Clear List")
            download_btn = gr.Button("⬇️ DOWNLOAD ALL MISSING", variant="primary", size="lg")

        # --- LÓGICAS Y EVENTOS ---
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

        def add_model(state, url, hf_file, folder, subfolder, civ_token, hf_token):
            if not url: return state, state_to_df(state)
            
            real_url = url
            filename = "Pending..."
            
            if hf_file:
                match = re.search(r"huggingface\.co/([^/]+/[^/?#]+)(?:/tree/([^/?#]+))?", url)
                if match:
                    repo_id = match.group(1)
                    branch = match.group(2) if match.group(2) else "main"
                    real_url = f"https://huggingface.co/{repo_id}/resolve/{branch}/{hf_file}"
                    filename = hf_file

            new_item = {
                "url": url, 
                "selected_url": real_url, 
                "filename": filename,
                "filesize": "-- MB",
                "folder": folder,
                "subfolder": subfolder,
                "status": "Checking..."
            }
            state.append(new_item)
            return state, state_to_df(state)

        add_btn.click(fn=add_model, inputs=[model_state, url_input, hf_file_dropdown, folder_input, subfolder_input, civitai_input, hf_input], outputs=[model_state, model_df])

        clear_btn.click(fn=lambda: ([], []), outputs=[model_state, model_df])

        def export_json_file(state):
            if not state: return gr.update(visible=False)
            tmp_path = os.path.join(tempfile.gettempdir(), "AcademiaSD_Models_Preset.json")
            export_data = [{"url": s["url"], "selected_url": s["selected_url"], "filename": s["filename"], "filesize": s["filesize"], "folder": s["folder"], "subfolder": s["subfolder"]} for s in state]
            with open(tmp_path, "w") as f:
                json.dump(export_data, f, indent=4)
            return gr.update(value=tmp_path, visible=True)

        export_btn.click(fn=export_json_file, inputs=[model_state], outputs=[export_file])

        def import_json_file(file_obj, current_state):
            if file_obj is None: return current_state, state_to_df(current_state)
            try:
                with open(file_obj.name, "r") as f:
                    data = json.load(f)
                
                new_state =[]
                for item in data:
                    # Al importar, si viene de ComfyUI (checkpoints, loras), lo mantenemos igual. 
                    # El motor se encargará de guardarlo en Forge automáticamente
                    new_state.append({
                        "url": item.get("url", ""),
                        "selected_url": item.get("selected_url", item.get("url", "")),
                        "filename": item.get("filename", "Pending..."),
                        "filesize": item.get("filesize", "-- MB"),
                        "folder": item.get("folder", "Stable-diffusion"),
                        "subfolder": item.get("subfolder", ""),
                        "status": "Not checked"
                    })
                return new_state, state_to_df(new_state)
            except:
                return current_state, state_to_df(current_state)

        import_json.upload(fn=import_json_file, inputs=[import_json, model_state], outputs=[model_state, model_df])

        def refresh_status(state, civ_token, hf_token):
            for item in state:
                real_url = item.get("selected_url")
                if not real_url: continue
                
                fname, fsize = get_file_info_from_url(real_url, civ_token, hf_token)
                
                if not fname:
                    item["status"] = "❌ Need API Key"
                else:
                    item["filename"] = fname
                    item["filesize"] = fsize
                    # Usamos el Traductor Universal para comprobar si existe en el disco duro real
                    target_dir = get_target_path(item["folder"], item["subfolder"])
                    full_path = os.path.join(target_dir, fname)
                    
                    if os.path.exists(full_path):
                        try:
                            local_size = os.path.getsize(full_path)
                            item["filesize"] = format_size(local_size)
                        except: pass
                        item["status"] = "✅ Ready"
                    else:
                        item["status"] = "🔴 Missing"
            
            return state, state_to_df(state)

        refresh_btn.click(fn=refresh_status, inputs=[model_state, civitai_input, hf_input], outputs=[model_state, model_df])
        add_btn.click(fn=refresh_status, inputs=[model_state, civitai_input, hf_input], outputs=[model_state, model_df])
        import_json.upload(fn=refresh_status, inputs=[model_state, civitai_input, hf_input], outputs=[model_state, model_df])


        def download_all(state, civ_token, hf_token):
            state, _ = refresh_status(state, civ_token, hf_token)
            yield state, state_to_df(state)

            for item in state:
                if item["status"] == "✅ Ready" or "Need API Key" in item["status"]:
                    continue 
                
                real_url = item["selected_url"]
                fname = item["filename"]
                # Usamos el Traductor Universal para guardar
                target_dir = get_target_path(item["folder"], item["subfolder"])
                os.makedirs(target_dir, exist_ok=True)
                
                file_path = os.path.join(target_dir, fname)
                temp_path = file_path + ".temp"

                item["status"] = "⏳ Starting..."
                yield state, state_to_df(state)

                try:
                    req_headers = get_headers_with_auth(real_url, civ_token, hf_token)
                    with requests.get(real_url, stream=True, allow_redirects=True, headers=req_headers) as r:
                        r.raise_for_status()
                        total_length = r.headers.get('content-length')
                        
                        if total_length is None:
                            total_length = 0
                        else:
                            total_length = int(total_length)
                        
                        downloaded = 0
                        with open(temp_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=1024*1024):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    if total_length > 0:
                                        pct = int((downloaded / total_length) * 100)
                                        item["status"] = f"⏳ {pct}%"
                                        yield state, state_to_df(state)
                    
                    os.replace(temp_path, file_path)
                    item["status"] = "✅ Ready"
                    try:
                        item["filesize"] = format_size(os.path.getsize(file_path))
                    except: pass
                    yield state, state_to_df(state)

                except Exception as e:
                    item["status"] = "❌ Error"
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    yield state, state_to_df(state)

        download_btn.click(fn=download_all, inputs=[model_state, civitai_input, hf_input], outputs=[model_state, model_df])

    return[(academia_tab, "Academia Downloader ⬇️", "academia_downloader_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)