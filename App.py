from flask import Flask, render_template, request
import cv2
import os
import pywt

app = Flask(__name__)
app.config['DEBUG'] = True
UPLOAD_FOLDER = 'static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def apply_filters(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None  # Gestion d'erreur si l'image n'est pas lisible

    results = []
    
    # Appliquer la transformation en ondelettes pour restaurer les détails
    coeffs = pywt.dwt2(img, 'bior1.3')
    cA, (cH, cV, cD) = coeffs
    reconstructed_img = pywt.idwt2((cA, (cH, cV, cD)), 'bior1.3')
    wavelet_result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"Transformation_Ondelettes_{os.path.basename(image_path)}")
    cv2.imwrite(wavelet_result_path, reconstructed_img)
    results.append(("Transformation en Ondelettes", wavelet_result_path))

    # Augmenter la saturation des couleurs
    enhanced_color_img = cv2.convertScaleAbs(reconstructed_img, alpha=1.2, beta=10)
    enhanced_color_result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"Image_Couleur_Augmentee_{os.path.basename(image_path)}")
    cv2.imwrite(enhanced_color_result_path, enhanced_color_img)
    results.append(("Augmentation des Couleurs", enhanced_color_result_path))

    # Appliquer le filtre de réduction de bruit de Wiener
    wiener_filtered = cv2.fastNlMeansDenoisingColored(enhanced_color_img, None, 10, 10, 7, 21)
    wiener_result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"Filtre_Wiener_{os.path.basename(image_path)}")
    cv2.imwrite(wiener_result_path, wiener_filtered)
    results.append(("Filtre de Wiener", wiener_result_path))
    
    # Tracer les contours pour améliorer les détails
    gray_img = cv2.cvtColor(wiener_filtered, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 100, 200)
    contour_result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"Contours_{os.path.basename(image_path)}")
    cv2.imwrite(contour_result_path, edges)
    results.append(("Tracé des Contours", contour_result_path))

    # Convertir l'image des contours améliorés en couleur et ajuster les dimensions/canaux
    color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    color_edges_resized = cv2.resize(color_edges, (img.shape[1], img.shape[0]))
    if color_edges_resized.shape[-1] != img.shape[-1]:
        color_edges_resized = cv2.cvtColor(color_edges_resized, cv2.COLOR_BGR2GRAY)
        color_edges_resized = cv2.cvtColor(color_edges_resized, cv2.COLOR_GRAY2BGR)

    # Appliquer un filtre de renforcement pour améliorer les détails
    enhanced_edges = cv2.detailEnhance(color_edges_resized, sigma_s=10, sigma_r=0.15)
    enhanced_edges_path = os.path.join(app.config['UPLOAD_FOLDER'], f"Contours_Ameliores_{os.path.basename(image_path)}")
    cv2.imwrite(enhanced_edges_path, enhanced_edges)
    results.append(("Amélioration des Contours", enhanced_edges_path))

    # Appliquer un filtre de netteté pour améliorer les détails
    sharpened_img = cv2.detailEnhance(wiener_filtered, sigma_s=10, sigma_r=0.15)
    sharpened_result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"Image_Nette_{os.path.basename(image_path)}")
    cv2.imwrite(sharpened_result_path, sharpened_img)
    results.append(("Amélioration de la Netteté", sharpened_result_path))

    # Ajuster la luminosité et le contraste
    alpha = 1  # Facteur de contraste
    beta = 10  # Ajustement de la luminosité
    adjusted_img = cv2.convertScaleAbs(sharpened_img, alpha=alpha, beta=beta)
    adjusted_result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"Image_Ajustee_{os.path.basename(image_path)}")
    cv2.imwrite(adjusted_result_path, adjusted_img)
    results.append(("Image Ajustée", adjusted_result_path))

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/apply_filters', methods=['POST'])
def apply_filters_route():
    if request.method == 'POST':
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename != '':
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
                image_file.save(image_path)

                results = apply_filters(image_path)
                if results is None:
                    return render_template('error.html', message='Erreur lors du traitement de l\'image.')
                print(results[1])
                return render_template('results.html',
                                       image_path=image_file.filename,
                                       results=results)
    
    return render_template('error.html', message='Aucune image téléchargée.')

if __name__ == '__main__':
    app.run(debug=True)
