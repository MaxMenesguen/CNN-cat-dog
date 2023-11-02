let modele;
let imageTensor;

async function chargerModele() {
    modele = await ort.InferenceSession.create('sortie_CNN_chat_chien.onnx');
    console.log('Modèle chargé');
}

function chargerImage(event) {
    const input = event.target;
    const reader = new FileReader();
    reader.onload = function(){
      const img = new Image();
      img.onload = function(){
          const canvasOriginal = document.getElementById('imageImportee');
          const canvasGris = document.getElementById('imageTransformee');
          const ctxOriginal = canvasOriginal.getContext('2d');
          const ctxGris = canvasGris.getContext('2d');

          // Calculer le ratio et définir les nouvelles dimensions
          const ratio = Math.min(200 / img.width, 200 / img.height);
          const nouveauWidth = img.width * ratio;
          const nouveauHeight = img.height * ratio;

          // Redimensionner les canvas
          canvasOriginal.width = 200;
          canvasOriginal.height = 200;
          canvasGris.width = 50;
          canvasGris.height = 50;

          // Centre l'image dans le canvas
          const x = (200 - nouveauWidth) / 2;
          const y = (200 - nouveauHeight) / 2;
          ctxOriginal.drawImage(img, x, y, nouveauWidth, nouveauHeight);

          // Dessiner l'image redimensionnée dans le canvas gris
          ctxGris.drawImage(img, 0, 0, 50, 50);
          const imageDataGris = ctxGris.getImageData(0, 0, 50, 50);
          const dataGris = new Float32Array(50 * 50);

          // Convertir en niveaux de gris et redimensionner
          for (let i = 0; i < imageDataGris.data.length; i += 4) {
            const pixelIndex = i / 4;
            // Convertir les valeurs RGB en une valeur en niveaux de gris
            const gray = 0.299 * imageDataGris.data[i] + 0.587 * imageDataGris.data[i + 1] + 0.114 * imageDataGris.data[i + 2];
            
            // Mettre à jour les canaux rouge, vert et bleu avec la valeur en niveaux de gris
            imageDataGris.data[i] = gray;
            imageDataGris.data[i + 1] = gray;
            imageDataGris.data[i + 2] = gray;
        
            // Stocker la valeur en niveaux de gris dans le tableau de données
            dataGris[pixelIndex] = gray / 255.0;
          }
        
          // Dessiner l'image en niveaux de gris sur le canvas
          ctxGris.putImageData(imageDataGris, 0, 0);
        
          imageTensor = new ort.Tensor('float32', dataGris, [1, 1, 50, 50]);
          console.log('Tenseur d\'entrée:', imageTensor);

        }
      img.src = reader.result;
    }
    reader.readAsDataURL(input.files[0]);
}

async function executerModele() {
    if (!modele || !imageTensor) {
      alert('Veuillez charger un modèle et une image.');
      return;
    }

    console.log('Shape de l\'image:', imageTensor.dims);

    const inputName = "input.1";
    const feeds = {};
    feeds[inputName] = imageTensor;

    const outputMap = await modele.run(feeds);
    const outputTensor = outputMap['25']; // Utilisez la notation d'accolade pour accéder au tenseur de sortie
    const predictions = outputTensor.data;
    console.log('Prédictions:', predictions);

    const resultat = document.getElementById('resultat');
    function round(number, decimals) {
      const factor = Math.pow(10, decimals);
      return Math.round(number * factor) / factor;
    }
    
    if (predictions[1] > predictions[0]) {
      const confidence = predictions[1];
      const roundedConfidence = round(confidence, 2) * 100;
      resultat.innerText = `C'est un chien ! Le réseau est sûr à ${roundedConfidence}%`;
    } else {
      const confidence = predictions[0];
      const roundedConfidence = round(confidence, 2) * 100;
      resultat.innerText = `C'est un chat ! Le réseau est sûr à ${roundedConfidence}%`;
    }
    
}

window.onload = chargerModele;
