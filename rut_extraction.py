import os
import re
import unicodedata
import easyocr
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from datetime import datetime
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

def log_message(message):
    """
    Imprime un mensaje con marca de tiempo.
    """
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def try_to_use_gpu():
    """
    Verifica la disponibilidad de una GPU y configura el lector OCR para usarla si está disponible.
    """
    global ocr_reader, use_gpu
    
    if torch.cuda.is_available():
        log_message(f"CUDA disponible. GPU activa: {torch.cuda.get_device_name(0)}")
        use_gpu = True
    else:
        log_message("CUDA no disponible. Se utilizará CPU.")
        use_gpu = False

    ocr_reader = easyocr.Reader(['es'], gpu=use_gpu)
    log_message(f"OCR utilizando {'GPU' if use_gpu else 'CPU'}.")

def liberar_memoria():
    """
    Libera la memoria de la GPU si está disponible.
    """
    torch.cuda.empty_cache()
    
def plot_image(img, title, title_color="black", subtitle=""):
    """
    Muestra una imagen con un título y subtítulo opcionales.
    """
    plt.imshow(img, cmap='gray')
    plt.suptitle(title, color=title_color)
    plt.title(subtitle)
    plt.show()

def combine_images(images):
    """
    Combina una lista de imágenes apilándolas verticalmente en una sola imagen.
    """
    max_width = max(img.shape[1] for img in images)
    resized_images = [cv2.resize(img, (max_width, img.shape[0])) for img in images]
    return cv2.vconcat(resized_images)

def format_rut(unformatted_rut):
    formatted_rut = unformatted_rut.lower().split('run', 1)[-1]
    formatted_rut = formatted_rut.replace('|', '1')
    formatted_rut = formatted_rut.replace('o', '0')
    formatted_rut = re.sub(r'[^0-9- ]', '', formatted_rut)
    formatted_rut = max(formatted_rut.split(' '), key=len).split('-')[0]
    if 6 <= len(formatted_rut) <= 10:
        return formatted_rut
    return None



def get_rut_zone_from_document(img):
    print("BUSCANDO...")

    # Cargar la plantilla (la palabra) en escala de grises
    plantilla_ruta = r"C:\Users\Basss\Desktop\EJEMPLOS_NAC\PLANTILLA.png"
    plantilla = cv2.imread(plantilla_ruta, 0)  # Cargar plantilla en escala de grises
    
    # Asegurarse de que la imagen objetivo también esté en escala de grises
    imagen_gris_objetivo = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar binarización (umbralización) para resaltar el texto
    _, plantilla_binaria = cv2.threshold(plantilla, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, imagen_binaria_objetivo = cv2.threshold(imagen_gris_objetivo, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Realizar el template matching con la imagen binarizada
    resultado = cv2.matchTemplate(imagen_binaria_objetivo, plantilla_binaria, cv2.TM_CCOEFF_NORMED)
    
    # Encontrar los valores mínimos y máximos, y las ubicaciones correspondientes
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resultado)

    # Obtener la posición superior izquierda de la mejor coincidencia
    esquina_superior_izquierda = max_loc
    # Obtener el tamaño de la plantilla
    ancho, alto = plantilla.shape[::-1]
    # Calcular la esquina inferior derecha de la coincidencia
    esquina_inferior_derecha = (esquina_superior_izquierda[0] + ancho, esquina_superior_izquierda[1] + alto)

    # Dibujar un rectángulo alrededor de la coincidencia en la imagen original
    cv2.rectangle(img, esquina_superior_izquierda, esquina_inferior_derecha, (0, 255, 0), 2)

    # Mostrar la imagen con la coincidencia resaltada
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Coincidencia encontrada con valor: {max_val:.2f}")
    plt.show()

# Ejemplo de uso:
# img = cv2.imread('ruta_a_tu_imagen.jpg')
# get_word_zone_from_document(img)


   
   

def get_rut_zones_from_ci(img, rotation_degree):
    """
    Busca posibles zonas de RUT en una imagen detectando rostros.
    Si no se encuentran zonas, rota la imagen y repite la búsqueda.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    rut_regions = []
    for (x, y, w, h) in faces:
        inicio_rut_y = max(0, y + h + h // 4)
        fin_rut_y = y + h * 2
        inicio_rut_x = max(0, x - w // 3)
        fin_rut_x = x + w + w // 3
        rut_region = img[inicio_rut_y:fin_rut_y, inicio_rut_x:fin_rut_x]
        if rut_region.size > 0:
            rut_regions.append(rut_region)
    
    if rut_regions:
        return rut_regions
    elif rotation_degree == 0:
        rotated_image = np.rot90(img, k=3)
        return get_rut_zones_from_ci(rotated_image, 90)
    elif rotation_degree == 90:
        rotated_image = np.rot90(img, k=2)
        return get_rut_zones_from_ci(rotated_image, 270)
    
    original_image = np.rot90(img, k=3)
    return get_rut_zone_from_document(original_image)

def search_ruts(pdf_path):
    """
    Busca posibles RUTs en un archivo PDF aplicando OCR en las zonas detectadas.
    """
    global ocr_reader, use_pytesseract
    
    images = convert_from_path(pdf_path)
    results_in_file = []
    
    for img in images:
        img_np = np.array(img)
        rut_zones = get_rut_zones_from_ci(img_np, 0)
    
        if rut_zones:
            for zone in rut_zones:
                if use_pytesseract:
                    img_pil = Image.fromarray(zone)
                    ocr_text = pytesseract.image_to_string(img_pil, lang='spa')
                else:
                    result_easy = ocr_reader.readtext(zone, detail=0)
                    ocr_text = " ".join(result_easy)
                if len(ocr_text) > 7:
                    results_in_file.append(ocr_text)
                    
    if results_in_file:
        for i in range(len(results_in_file)):
            results_in_file[i] = format_rut(results_in_file[i])
        results_in_file = list(filter(bool, results_in_file))
        
    liberar_memoria()
    return results_in_file if results_in_file else None

def get_ci_file(files):
    """
    Busca un archivo en una lista cuyo nombre contenga términos relacionados con documentos de identidad.
    """
    pattern = re.compile(r'(?<!\w)ci(?!\w)', re.IGNORECASE)
    regular_expressions = [r'c\.i', r'\brun\b', r'\brut\b', r'\bcedula\b', r'\bidentidad\b', r'\bcarnet\b', r'\bsii\b', r'\brol\b']
    
    for pdf in files:
        filename = unicodedata.normalize('NFKD', os.path.basename(pdf).lower())
        filename = ''.join(c for c in filename if not unicodedata.combining(c))
        if re.search(pattern, filename):
            return pdf
        for regex in regular_expressions:
            if re.search(regex, filename):
                return pdf
    return None

def process_subdirectory(subdirectory):
    """
    Procesa un subdirectorio buscando archivos PDF que contengan información de CI.
    """
    global subdirectories_with_ci_file, subdirectories_without_ci_file, subdirectories_count, solicitudes_con_rut
    subdirectory_name = os.path.basename(subdirectory)
    log_message(f"🔎 {subdirectory_name}: PROCESANDO ({subdirectories_with_ci_file + subdirectories_without_ci_file + 1}/{subdirectories_count})...")
    
    pdf_files = [os.path.join(root, file) for root, _, files in os.walk(subdirectory) for file in files if file.lower().endswith('.pdf')]
    ci_file = get_ci_file(pdf_files)
    
    if ci_file:
        subdirectories_with_ci_file += 1
        founded_ruts = search_ruts(ci_file)
        if founded_ruts:
            log_message(f"✅ {subdirectory_name}: {str(founded_ruts)}")
            solicitudes_con_rut += 1
        else:
            log_message(f"❌ {subdirectory_name}: NO SE ENCONTRARON RUTS")
    else:
        subdirectories_without_ci_file += 1
        log_message(f"🚫 {subdirectory_name}: NO SE ENCONTRÓ EL ARCHIVO \t")

if __name__ == "__main__":
    
    test_directory = r"ruta"
    
    use_pytesseract = False  # Variable que indica si se usará Tesseract para el OCR.
    use_gpu = False  # Variable que indica si se usará GPU para el procesamiento OCR.
    ocr_reader = None  # Objeto del lector OCR que se inicializará más adelante.
    subdirectories_with_ci_file = 0  # Cantidad de subdirectorios que contienen posibles archivos CI.pdf.
    subdirectories_without_ci_file = 0  # Cantidad de subdirectorios en los que no se encontraron archivos CI.pdf.
    solicitudes_con_rut = 0  # Cantidad de solicitudes en las que se encontró un posible RUT.
    subdirectories_count = 0  # Cantidad total de subdirectorios que se procesarán.


    try_to_use_gpu()
    
    subdirectories = [os.path.join(test_directory, subdirectory_name.name) for subdirectory_name in os.scandir(test_directory) if subdirectory_name.is_dir(follow_symlinks=False)]
    subdirectories_count = len(subdirectories)
    
    for subdirectory in subdirectories:
        process_subdirectory(subdirectory)

    """
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_subdirectory, subdirectory) for subdirectory in subdirectories]
        for future in as_completed(futures):
            future.result()
    """
                
    log_message(f"{subdirectories_with_ci_file}/{subdirectories_with_ci_file + subdirectories_without_ci_file} CI encontradas. Faltan {subdirectories_without_ci_file}.")
    log_message(f"Se identificaron ruts en {solicitudes_con_rut} archivos de los {subdirectories_with_ci_file} probados.")

