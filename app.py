import os
import uuid
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEXTURE_FOLDER'] = 'static/textures'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Classes ampliadas (Pascal VOC classes)
CLASSES = {
    0: 'background', 7: 'car', 8: 'cat', 12: 'dog', 15: 'person', 
    17: 'horse', 19: 'sheep', 21: 'cow', 24: 'motorcycle', 41: 'umbrella'
}

# Carregar modelo pré-treinado
model = deeplabv3_resnet101(pretrained=True)
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)

def improved_color_mask(image, click_points, tolerance=30):
    """Função aprimorada para detecção de cores semelhantes usando limiarização."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    for (x, y) in click_points:
        # Verificar se o ponto clicado está dentro dos limites da imagem
        if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
            continue
        
        # Obter a cor do ponto clicado
        target_color = hsv[y, x]
        
        # Definir limites de tolerância para a cor
        lower = np.array([
            max(0, int(target_color[0]) - tolerance),
            max(0, int(target_color[1]) - tolerance),
            max(0, int(target_color[2]) - tolerance)
        ], dtype=np.uint8)
        
        upper = np.array([
            min(179, int(target_color[0]) + tolerance),
            min(255, int(target_color[1]) + tolerance),
            min(255, int(target_color[2]) + tolerance)
        ], dtype=np.uint8)
        
        # Criar máscara para a área clicada usando limiarização
        mask = cv2.inRange(hsv, lower, upper)
        
        # Adicionar a região à máscara combinada
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    return combined_mask

def segment_image(image_path):
    """Segmentação de imagem com suporte para múltiplas classes."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image.shape[:2]
    
    resized_image = cv2.resize(image, (520, 520))
    input_tensor = preprocess_image(resized_image)
    
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    
    # Máscara combinada para múltiplas classes relevantes
    combined_mask = np.zeros_like(output_predictions)
    for class_id in [7, 8, 12, 15, 17, 21, 24]:  # Classes de objetos
        combined_mask = np.logical_or(combined_mask, output_predictions == class_id)
    
    mask = combined_mask.astype(np.uint8) * 255
    mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    
    return image, mask

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if not (file and allowed_file(file.filename)):
        return redirect(request.url)

    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)
    
    try:
        original_image, mask = segment_image(upload_path)
        upload_id = str(uuid.uuid4())
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], upload_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(temp_dir, 'original.png'), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(temp_dir, 'mask.png'), mask)
        
        return redirect(url_for('select_color', upload_id=upload_id))
    
    except Exception as e:
        print(f'Error: {str(e)}')
        return redirect(request.url)

@app.route('/select_color/<upload_id>')
def select_color(upload_id):
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], upload_id)
    if not os.path.exists(temp_dir):
        return "Arquivo não encontrado", 404
    
    textures = os.listdir(app.config['TEXTURE_FOLDER'])
    original_url = url_for('serve_upload', filename=f'{upload_id}/original.png')
    return render_template('select_color.html', 
                         upload_id=upload_id,
                         original_url=original_url,
                         textures=textures)

@socketio.on('update_preview')
def handle_update_preview(data):
    upload_id = data['upload_id']
    click_points = data['click_points']
    tolerance = data.get('tolerance', 30)
    
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], upload_id)
    original_path = os.path.join(temp_dir, 'original.png')
    mask_path = os.path.join(temp_dir, 'mask.png')
    
    original_image = cv2.imread(original_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    object_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Criar máscara de cor melhorada para múltiplos pontos
    color_mask = improved_color_mask(original_image, click_points, tolerance=tolerance)
    
    # Combinar máscaras (se objeto detectado)
    if np.any(object_mask):
        combined_mask = cv2.bitwise_and(color_mask, object_mask)
    else:
        combined_mask = color_mask
    
    # Salvar máscara de pré-visualização
    preview_filename = f'preview_{upload_id}.png'
    preview_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_id, preview_filename)
    cv2.imwrite(preview_path, combined_mask)
    
    preview_url = url_for('serve_upload', filename=f'{upload_id}/{preview_filename}')
    emit('preview_updated', {'preview_url': preview_url})

@app.route('/apply_texture', methods=['POST'])
def apply_texture_route():
    data = request.get_json()
    upload_id = data['upload_id']
    click_points = data['click_points']
    texture_name = data['texture']
    opacity = float(data['opacity'])  # Opacidade da textura (0 a 1)
    
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], upload_id)
    original_path = os.path.join(temp_dir, 'original.png')
    mask_path = os.path.join(temp_dir, 'mask.png')
    texture_path = os.path.join(app.config['TEXTURE_FOLDER'], texture_name)
    
    original_image = cv2.imread(original_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    object_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Criar máscara de cor melhorada para múltiplos pontos
    color_mask = improved_color_mask(original_image, click_points)
    
    # Combinar máscaras (se objeto detectado)
    if np.any(object_mask):
        combined_mask = cv2.bitwise_and(color_mask, object_mask)
    else:
        combined_mask = color_mask
    
    # Aplicar textura com suavização
    texture = cv2.imread(texture_path)
    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
    texture = cv2.resize(texture, (original_image.shape[1], original_image.shape[0]))
    
    # Suavizar bordas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.GaussianBlur(combined_mask, (5,5), 0)
    
    # Converter máscara para 3 canais e normalizar
    mask_float = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    
    # Aplicar textura com opacidade ajustável
    result = (original_image * (1 - mask_float * opacity) + texture * mask_float * opacity).astype(np.uint8)
    
    # Salvar resultado
    result_filename = f'result_{upload_id}.png'
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    result_url = url_for('static', filename=f'results/{result_filename}')
    return jsonify({
        'success': True,
        'result_url': result_url
    })

@app.route('/result')
def show_result():
    result_url = request.args.get('image')
    return render_template('result.html', result_image=result_url)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    socketio.run(app, debug=True)