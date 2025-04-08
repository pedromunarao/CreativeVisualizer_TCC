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
    process_image(filepath, processed_filepath)

    return redirect(url_for('select_color', upload_id=upload_id))

def process_image(image_path, output_path, bg_color=(0, 0, 0)):
    original = Image.open(image_path).convert("RGBA")
    no_bg = remove(original)
    new_bg = Image.new("RGBA", no_bg.size, bg_color + (255,))
    final_image = Image.alpha_composite(new_bg, no_bg)
    final_image.convert("RGB").save(output_path, "PNG")

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

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_id, 'processed.png')
        mask_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_id, 'mask.png')

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Criar máscara de fundo preto
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([10, 10, 10], dtype=np.uint8)
        background_mask = cv2.inRange(img, lower_black, upper_black)
        foreground_mask = cv2.bitwise_not(background_mask)

        flood_mask = np.zeros((h, w), dtype=np.uint8)

        for (x, y) in points:
            seed_point = (x, y)
            fill_color = (0, 0, 255)
            lo_diff = (tolerance, tolerance, tolerance)
            up_diff = (tolerance, tolerance, tolerance)
            flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)

            img_copy = img.copy()
            cv2.floodFill(img_copy, mask, seedPoint=seed_point, newVal=fill_color,
                          loDiff=lo_diff, upDiff=up_diff, flags=flags)

            mask_area = mask[1:h + 1, 1:w + 1]

            # Ignorar fundo preto removido
            mask_area = cv2.bitwise_and(mask_area, foreground_mask)

            flood_mask = cv2.bitwise_or(flood_mask, mask_area)

        cv2.imwrite(mask_path, flood_mask)
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

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        texture = cv2.imread(texture_path)

        texture = cv2.resize(texture, (img.shape[1], img.shape[0]))
        mask_float = mask.astype(float) / 255

        result = cv2.addWeighted(img, 1 - opacity, texture, opacity, 0)

        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - mask_float) + result[:, :, c] * mask_float

        result_filename = f'result_{upload_id}.png'
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, img)

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
