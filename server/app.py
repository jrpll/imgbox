from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import io
from finedits import FINEdits
from diffusers import StableDiffusion3Pipeline
import torch
from PIL import Image
from flask import send_from_directory

app = Flask(__name__)
app.config['PROPAGATE_EXCEPTIONS'] = True 
CORS(app)

# Add a folder to temporarily store uploaded files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

token = os.getenv('HUGGING_FACE_TOKEN')
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.bfloat16, token=token).to("cuda")
image_editor = FINEdits(pipe)

@app.route('/edit', methods=['POST'])
def edit():
    try:
        strength = float(request.form.get("slider", "0.9")) #le truc pour un slider
        num_inversion_steps = image_editor.num_inversion_steps
        print("num inversion steps",num_inversion_steps)
        num_skipped_steps = num_inversion_steps - int(strength*num_inversion_steps)
        print("nst",num_skipped_steps)
        img_prompt = request.form.get('text1', '')
        target_prompt = request.form.get('text2', '')
        neg_prompt = img_prompt.replace(target_prompt,"").strip(", ")
        print("editing")
        print("target",target_prompt)
        print("neg prompt",neg_prompt)
        edited_img = image_editor.edit(
            prompt = target_prompt,
            neg_prompt = neg_prompt,
            num_skipped_steps = num_skipped_steps
        )
        print("done editing")
        img_io = io.BytesIO()
        edited_img.save(img_io, "JPEG")
        img_io.seek(0)    
        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        #return jsonify({'error': str(e)}), 500
        import traceback
        print("Error during separate_concepts:")
        print(traceback.format_exc())
        raise e

@app.route('/generate', methods=['POST'])
def generate():
    print("HELLO")
    try:
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image = request.files['image']
        img_prompt = request.form.get('text1', '')
        target_prompt = request.form.get('text2', '')
        neg_prompt = img_prompt.replace(target_prompt,"").strip(", ")
        
        print(f"Received: image={image.filename}, text1={img_prompt}, text2={target_prompt}")
        
        filename = secure_filename(image.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image.save(filepath)
        image = Image.open(image).resize((1024,1024)).convert("RGB")
        print("starting fine tuning")
        print("img prompt",img_prompt)
        image_editor.ft_invert(
            img = image,
            prompt = img_prompt,
            num_train_steps = 100,
            fine_tune = True,
            num_inversion_steps = 50
        )
        print("done fine tuning")
        print("tgt",target_prompt)
        print("neg",neg_prompt)
        edited_img = image_editor.edit(
            prompt = target_prompt,
            neg_prompt = neg_prompt
        )
        print("done")
        img_io = io.BytesIO()
        edited_img.save(img_io, "JPEG")
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')
    
    except Exception as e:
        #return jsonify({'error': str(e)}), 500
        import traceback
        print("Error during separate_concepts:")
        print(traceback.format_exc())
        raise e
    
@app.route('/')
def serve_frontend():
    return send_from_directory('../client/dist', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../client/dist', path)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
