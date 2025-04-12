from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import uuid
import cv2
import numpy as np
from flask_socketio import SocketIO, emit
from rembg import remove
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEXTURE_FOLDER'] = 'static/textures'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['SECRET_KEY'] = 'sua_chave_secreta_aqui'

socketio = SocketIO(app, cors_allowed_origins="*")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def resize_image(image_path, max_size=2000):
    """Redimensiona a imagem se for maior que max_size (em pixels) mantendo a proporção"""
    img = Image.open(image_path)
    width, height = img.size
    
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
            
        img = img.resize((new_width, new_height), Image.LANCZOS)
        img.save(image_path)
        return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<upload_id>/<filename>')
def serve_upload(upload_id, filename):
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], upload_id), filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if not (file and allowed_file(file.filename)):
        return redirect(request.url)

    filename = secure_filename(file.filename)
    upload_id = str(uuid.uuid4())
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], upload_id)
    os.makedirs(temp_dir, exist_ok=True)

    filepath = os.path.join(temp_dir, 'original.png')
    processed_filepath = os.path.join(temp_dir, 'processed.png')

    file.save(filepath)
    
    # Redimensionar imagem se for muito grande
    resize_image(filepath)

    remove_bg = request.form.get('remove_bg') == 'on'
    process_image(filepath, processed_filepath, remove_background=remove_bg)

    return redirect(url_for('select_color', upload_id=upload_id))

def process_image(image_path, output_path, remove_background=True):
    original = Image.open(image_path).convert("RGBA")

    if remove_background:
        no_bg = remove(original)
        transparent_bg = Image.new("RGBA", no_bg.size, (0, 0, 0, int(255 * 0.2)))
        final_image = Image.alpha_composite(transparent_bg, no_bg)
    else:
        final_image = original

    final_image.save(output_path, "PNG")

@app.route('/select_color/<upload_id>')
def select_color(upload_id):
    textures = os.listdir(app.config['TEXTURE_FOLDER'])
    processed_url = url_for('serve_upload', upload_id=upload_id, filename='processed.png')
    return render_template('select_color.html',
                           upload_id=upload_id,
                           processed_url=processed_url,
                           textures=textures)

@app.route('/result')
def show_result():
    result_url = request.args.get('image')
    return render_template('result.html', result_image=result_url)

@socketio.on('select_area')
def handle_area_selection(data):
    try:
        upload_id = data['upload_id']
        points = data['points']
        tolerance = int(data['tolerance'])
        mode = data.get('mode', 'select')

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_id, 'processed.png')
        mask_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_id, 'mask.png')

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        for (x, y) in points:
            reference_color = img[y, x].astype(np.int16)
            color_diff = np.abs(img.astype(np.int16) - reference_color)
            color_distance = np.sum(color_diff, axis=2)
            selected_area = (color_distance <= tolerance * 3).astype(np.uint8) * 255

            if mode == 'select':
                mask = cv2.bitwise_or(mask, selected_area)
            elif mode == 'remove':
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(selected_area))

        cv2.imwrite(mask_path, mask)
        emit('selection_done', {'status': 'success', 'mask_url': f'/uploads/{upload_id}/mask.png'})

    except Exception as e:
        emit('selection_error', {'error': str(e)})

@socketio.on('brush_area')
def handle_brush_area(data):
    try:
        upload_id = data['upload_id']
        points = data['points']
        radius = int(data['radius'])
        mode = data.get('mode', 'brush')

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_id, 'processed.png')
        mask_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_id, 'mask.png')

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        for (x, y) in points:
            if mode == 'brush':
                cv2.circle(mask, (x, y), radius, 255, -1)
            elif mode == 'remove':
                cv2.circle(mask, (x, y), radius, 0, -1)

        cv2.imwrite(mask_path, mask)
        emit('selection_done', {'status': 'success', 'mask_url': f'/uploads/{upload_id}/mask.png'})

    except Exception as e:
        emit('selection_error', {'error': str(e)})

@socketio.on('apply_texture')
def handle_texture_application(data):
    try:
        upload_id = data['upload_id']
        texture_name = data['texture']
        opacity = float(data['opacity'])

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_id, 'processed.png')
        mask_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_id, 'mask.png')
        texture_path = os.path.join(app.config['TEXTURE_FOLDER'], texture_name)

        # Redimensionar textura se necessário
        resize_image(texture_path, 4000)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        texture = cv2.imread(texture_path)

        texture = cv2.resize(texture, (img.shape[1], img.shape[0]))

        # Cria textura parcialmente opaca com base na máscara
        texture_with_opacity = cv2.addWeighted(texture, opacity, img, 1 - opacity, 0)
        result = img.copy()
        for c in range(3):  # aplica textura nas áreas da máscara
            result[:, :, c] = np.where(mask == 255, texture_with_opacity[:, :, c], img[:, :, c])

        result_filename = f'result_{upload_id}.png'
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, result)

        emit('texture_applied', {
            'status': 'success',
            'result_url': f'/static/results/{result_filename}'
        })
    except Exception as e:
        emit('application_error', {'error': str(e)})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    os.makedirs(app.config['TEXTURE_FOLDER'], exist_ok=True)
    socketio.run(app, debug=True)

