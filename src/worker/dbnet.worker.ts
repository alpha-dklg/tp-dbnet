// @ts-ignore - OpenCV.js and ONNX Runtime are loaded via CDN
importScripts('https://docs.opencv.org/4.x/opencv.js');
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

// Déclarations globales pour TypeScript
declare const cv: any;
declare const ort: any;

// ============================================================================
// TYPES & INTERFACES
// ============================================================================

interface LineRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface PreprocessResult {
  tensor: any; // ort.Tensor
  scale: number;
  width: number;
  height: number;
}

interface InferenceResult {
  probMap: Float32Array;
  width: number;
  height: number;
}

interface PostprocessResult {
  lines: LineRect[];
}

interface WorkerMessage {
  msg: 'detect';
  payload: {
    image: ImageData;
  };
}

interface WorkerResponse {
  msg: 'lines' | 'error';
  payload: PostprocessResult | { stage: string; message: string; err?: any };
}

// ============================================================================
// CONFIGURATION ORT
// ============================================================================

// @ts-ignore - Configuration ORT pour worker
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.logLevel = 'warning';

// ============================================================================
// CONSTANTES
// ============================================================================

// Configuration du modèle
const MODEL_INPUT_NAME = 'x';
const MODEL_OUTPUT_NAME = 'fetch_name_0';

// Déterminer le chemin de base dynamiquement depuis l'URL du worker
// En dev: '/' et en prod (GitHub Pages): '/tp-dbnet/'
const getBasePath = () => {
  // Dans un worker, on peut utiliser self.location.pathname
  // Vite transforme import.meta.url correctement pour les workers
  const pathname = (self as any).location?.pathname || '';
  // Si on est dans un sous-dossier GitHub Pages
  if (pathname.includes('/tp-dbnet/')) {
    return '/tp-dbnet';
  }
  return '';
};

const BASE_PATH = getBasePath();
const MODEL_PATH = `${BASE_PATH}/models/det_model.onnx`;

// Paramètres de preprocessing
const RESIZE_MAX_SIDE = 960;
const DBNET_PADDING_MULTIPLE = 32; // DBNet attend des dimensions multiples de 32
const PIXEL_MAX_VALUE = 255;
const PIXEL_NORMALIZATION_DIVISOR = 255;

// Normalisation ImageNet/PaddleOCR
const NORMALIZATION_MEAN = [0.485, 0.456, 0.406];
const NORMALIZATION_STD = [0.229, 0.224, 0.225];

// Paramètres de postprocessing
const THRESHOLD = 0.3;
const MIN_BOX_WIDTH = 5;
const MIN_BOX_HEIGHT = 5;
const MERGE_TOL_FACTOR = 0.15;
const MERGE_PADDING = 4;

// Couleurs pour padding (noir)
const PADDING_COLOR = new cv.Scalar(0, 0, 0, 255);

// ============================================================================
// VARIABLES GLOBALES
// ============================================================================

let session: any | null = null;

// ============================================================================
// UTILITAIRES
// ============================================================================

/**
 * Calcule les dimensions paddées pour être multiples de la valeur spécifiée
 */
function calculatePaddedDimensions(width: number, height: number, multiple: number): { width: number; height: number } {
  return {
    width: Math.ceil(width / multiple) * multiple,
    height: Math.ceil(height / multiple) * multiple,
  };
}

/**
 * Calcule le facteur d'échelle pour redimensionner une image
 */
function calculateScale(originalWidth: number, originalHeight: number, maxSide: number): number {
  return Math.min(1, maxSide / Math.max(originalWidth, originalHeight));
}

/**
 * Calcule le centre vertical d'une boîte
 */
function getVerticalCenter(box: LineRect): number {
  return box.y + box.h / 2;
}

// ============================================================================
// ÉTAPE 1 : PRÉPROCESSING
// ============================================================================

/**
 * Redimensionne une image avec OpenCV
 */
function resizeImage(imageData: ImageData, targetWidth: number, targetHeight: number): any {
  // @ts-ignore - OpenCV types
  const sourceMat = cv.matFromImageData(imageData);
  const destinationMat = new cv.Mat();
  const targetSize = new cv.Size(targetWidth, targetHeight);
  cv.resize(sourceMat, destinationMat, targetSize, 0, 0, cv.INTER_AREA);
  return { sourceMat, destinationMat };
}

/**
 * Ajoute du padding noir autour d'une image pour atteindre les dimensions cibles
 */
function padImage(sourceMat: any, targetWidth: number, targetHeight: number, sourceWidth: number, sourceHeight: number): any {
  // @ts-ignore - OpenCV types
  const paddedMat = new cv.Mat(targetHeight, targetWidth, sourceMat.type(), PADDING_COLOR);
  const roiRect = new cv.Rect(0, 0, sourceWidth, sourceHeight);
  const roi = paddedMat.roi(roiRect);
  sourceMat.copyTo(roi);
  roi.delete();
  return paddedMat;
}

/**
 * Normalise les pixels RGB selon les statistiques ImageNet
 */
function normalizePixels(paddedMat: any, targetWidth: number, targetHeight: number): Float32Array {
  const tensorSize = 1 * 3 * targetHeight * targetWidth;
  const normalizedData = new Float32Array(tensorSize);

  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      // @ts-ignore - OpenCV types
      const pixelPtr = paddedMat.ucharPtr(y, x);
      const red = pixelPtr[0] / PIXEL_NORMALIZATION_DIVISOR;
      const green = pixelPtr[1] / PIXEL_NORMALIZATION_DIVISOR;
      const blue = pixelPtr[2] / PIXEL_NORMALIZATION_DIVISOR;

      const redIndex = 0 * targetHeight * targetWidth + y * targetWidth + x;
      const greenIndex = 1 * targetHeight * targetWidth + y * targetWidth + x;
      const blueIndex = 2 * targetHeight * targetWidth + y * targetWidth + x;

      normalizedData[redIndex] = (red - NORMALIZATION_MEAN[0]) / NORMALIZATION_STD[0];
      normalizedData[greenIndex] = (green - NORMALIZATION_MEAN[1]) / NORMALIZATION_STD[1];
      normalizedData[blueIndex] = (blue - NORMALIZATION_MEAN[2]) / NORMALIZATION_STD[2];
    }
  }

  return normalizedData;
}

/**
 * Transforme une ImageData en tensor normalisé pour DBNet
 * @param img - ImageData à préprocesser
 * @param maxSide - Taille maximale du côté le plus long (défaut: 960)
 * @returns Tensor préprocessé avec facteur d'échelle et dimensions
 */
function preprocess(img: ImageData, maxSide: number = RESIZE_MAX_SIDE): PreprocessResult {
  // Validation de l'entrée
  if (!img || !img.data || img.width <= 0 || img.height <= 0) {
    throw new Error('Invalid ImageData: image must have valid dimensions and data');
  }

  const { width, height } = img;

  // 1. Calcul des dimensions redimensionnées
  const scale = calculateScale(width, height, maxSide);
  const resizedWidth = Math.round(width * scale);
  const resizedHeight = Math.round(height * scale);

  // 2. Calcul des dimensions paddées (multiples de 32)
  const { width: paddedWidth, height: paddedHeight } = calculatePaddedDimensions(
    resizedWidth,
    resizedHeight,
    DBNET_PADDING_MULTIPLE
  );

  // 3. Redimensionnement avec OpenCV
  const { sourceMat, destinationMat } = resizeImage(img, resizedWidth, resizedHeight);

  // 4. Padding de l'image
  const paddedMat = padImage(destinationMat, paddedWidth, paddedHeight, resizedWidth, resizedHeight);

  // 5. Normalisation des pixels
  const normalizedData = normalizePixels(paddedMat, paddedWidth, paddedHeight);

  // 6. Création du tensor ONNX
  const tensor = new ort.Tensor('float32', normalizedData, [1, 3, paddedHeight, paddedWidth]);

  // 7. Nettoyage des ressources OpenCV
  sourceMat.delete();
  destinationMat.delete();
  paddedMat.delete();

  return {
    tensor,
    scale,
    width: resizedWidth,
    height: resizedHeight,
  };
}

// ============================================================================
// ÉTAPE 2 : INFERENCE
// ============================================================================

/**
 * Charge le modèle ONNX (une seule fois, en cache)
 */
async function loadModel(): Promise<any> {
  if (!session) {
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ['wasm'],
    });
  }
  return session;
}

/**
 * Exécute le modèle DBNet pour produire une heatmap de probabilités
 * @param tensor - Tensor préprocessé
 * @returns Heatmap de probabilités avec dimensions
 */
async function runInference(tensor: any): Promise<InferenceResult> {
  if (!tensor) {
    throw new Error('Invalid tensor: tensor is required for inference');
  }

  const modelSession = await loadModel();

  // Tentative avec le nom d'entrée configuré
  const feeds: Record<string, any> = {
    [MODEL_INPUT_NAME]: tensor,
  };

  let output: any;
  try {
    output = await modelSession.run(feeds);
  } catch (error) {
    // Fallback: essayer avec le nom 'images'
    const fallbackFeeds: Record<string, any> = { images: tensor };
    try {
      output = await modelSession.run(fallbackFeeds);
    } catch (fallbackError) {
      throw new Error(
        `Failed to run inference with both '${MODEL_INPUT_NAME}' and 'images' inputs. ` +
        `Original error: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }

  // Extraction du tenseur de sortie
  const outputTensor = output[MODEL_OUTPUT_NAME] ?? output[Object.keys(output)[0]];

  if (!outputTensor?.data || !outputTensor?.dims) {
    throw new Error(
      `Invalid output tensor. Available keys: ${Object.keys(output).join(', ')}`
    );
  }

  const probMap = outputTensor.data as Float32Array;
  const outputWidth = outputTensor.dims[outputTensor.dims.length - 1] as number;
  const outputHeight = outputTensor.dims[outputTensor.dims.length - 2] as number;

  return {
    probMap,
    width: outputWidth,
    height: outputHeight,
  };
}

// ============================================================================
// ÉTAPE 3 : POSTPROCESSING
// ============================================================================

/**
 * Fusionne plusieurs boîtes en une seule boîte englobante avec padding
 */
function mergeBoxesIntoLine(boxes: LineRect[]): LineRect {
  if (boxes.length === 0) {
    throw new Error('Cannot merge empty box array');
  }

  const leftX = Math.min(...boxes.map((box) => box.x));
  const topY = Math.min(...boxes.map((box) => box.y));
  const rightX = Math.max(...boxes.map((box) => box.x + box.w));
  const bottomY = Math.max(...boxes.map((box) => box.y + box.h));

  // Application du padding
  const paddedLeftX = Math.max(0, leftX - MERGE_PADDING);
  const paddedTopY = Math.max(0, topY - MERGE_PADDING);
  const paddedRightX = rightX + MERGE_PADDING;
  const paddedBottomY = bottomY + MERGE_PADDING;

  return {
    x: paddedLeftX,
    y: paddedTopY,
    w: paddedRightX - paddedLeftX,
    h: paddedBottomY - paddedTopY,
  };
}

/**
 * Transforme la heatmap de probabilités en lignes de texte détectées
 * @param probMap - Heatmap de probabilités du modèle
 * @param width - Largeur de la heatmap
 * @param height - Hauteur de la heatmap
 * @param scaleBack - Facteur d'échelle pour remettre à l'échelle originale
 * @returns Lignes de texte détectées avec coordonnées
 */
async function postprocess(
  probMap: Float32Array,
  width: number,
  height: number,
  scaleBack: number
): Promise<PostprocessResult> {
  // Validation des entrées
  if (!probMap || probMap.length === 0) {
    throw new Error('Invalid probability map: probMap must not be empty');
  }
  if (width <= 0 || height <= 0) {
    throw new Error(`Invalid dimensions: width=${width}, height=${height}`);
  }
  if (scaleBack <= 0) {
    throw new Error(`Invalid scale: scaleBack=${scaleBack} must be positive`);
  }

  // @ts-ignore - OpenCV types
  const probabilityMat = cv.matFromArray(height, width, cv.CV_32F, probMap);
  const probability8Bit = new cv.Mat();
  const binaryMat = new cv.Mat();

  // Normalisation et conversion en 8 bits
  // @ts-ignore - OpenCV types
  const scaleFactor = new cv.Mat(probabilityMat.rows, probabilityMat.cols, cv.CV_32F, new cv.Scalar(PIXEL_MAX_VALUE));
  cv.multiply(probabilityMat, scaleFactor, probabilityMat);
  probabilityMat.convertTo(probability8Bit, cv.CV_8U);

  // Seuillage binaire
  const thresholdValue = PIXEL_MAX_VALUE * THRESHOLD;
  cv.threshold(probability8Bit, binaryMat, thresholdValue, PIXEL_MAX_VALUE, cv.THRESH_BINARY);

  // Extraction des contours
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(binaryMat, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  // Conversion des contours en boîtes
  const boxes: LineRect[] = [];
  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i);
    const boundingRect = cv.boundingRect(contour);

    // Filtrage des boîtes trop petites
    if (boundingRect.width < MIN_BOX_WIDTH || boundingRect.height < MIN_BOX_HEIGHT) {
      contour.delete();
      continue;
    }

    // Remise à l'échelle des coordonnées
    boxes.push({
      x: Math.round(boundingRect.x / scaleBack),
      y: Math.round(boundingRect.y / scaleBack),
      w: Math.round(boundingRect.width / scaleBack),
      h: Math.round(boundingRect.height / scaleBack),
    });

    contour.delete();
  }

  // Regroupement des boîtes par lignes
  // Tri par centre vertical
  boxes.sort((a, b) => getVerticalCenter(a) - getVerticalCenter(b));

  const lines: LineRect[] = [];
  if (boxes.length > 0) {
    // Calcul de la tolérance verticale basée sur la hauteur moyenne
    const averageHeight = boxes.reduce((sum, box) => sum + box.h, 0) / boxes.length;
    const verticalTolerance = MERGE_TOL_FACTOR * averageHeight;

    let currentLineBoxes: LineRect[] = [boxes[0]];

    for (let i = 1; i < boxes.length; i++) {
      const currentBox = boxes[i];
      const lastBoxInLine = currentLineBoxes[currentLineBoxes.length - 1];

      const verticalDistance = Math.abs(
        getVerticalCenter(currentBox) - getVerticalCenter(lastBoxInLine)
      );

      if (verticalDistance <= verticalTolerance) {
        // Même ligne : ajouter à la ligne courante
        currentLineBoxes.push(currentBox);
      } else {
        // Nouvelle ligne : fusionner et sauvegarder la ligne précédente
        currentLineBoxes.sort((a, b) => a.x - b.x);
        lines.push(mergeBoxesIntoLine(currentLineBoxes));
        currentLineBoxes = [currentBox];
      }
    }

    // Ne pas oublier la dernière ligne
    if (currentLineBoxes.length > 0) {
      currentLineBoxes.sort((a, b) => a.x - b.x);
      lines.push(mergeBoxesIntoLine(currentLineBoxes));
    }
  }

  // Nettoyage des ressources OpenCV
  probabilityMat.delete();
  probability8Bit.delete();
  binaryMat.delete();
  contours.delete();
  hierarchy.delete();

  return { lines };
}

// ============================================================================
// ORCHESTRATION : Point d'entrée du worker
// ============================================================================

/**
 * Orchestre les trois étapes : preprocessing → inference → postprocessing
 */
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const { msg, payload } = event.data;

  if (msg === 'detect') {
    try {
      // Validation de l'entrée
      if (!payload?.image) {
        throw new Error('Missing image in payload');
      }

      // ÉTAPE 1 : Préprocessing
      const { tensor, scale } = preprocess(payload.image);

      // ÉTAPE 2 : Inference
      const { probMap, width: probWidth, height: probHeight } = await runInference(tensor);

      // ÉTAPE 3 : Postprocessing
      const result = await postprocess(probMap, probWidth, probHeight, scale);

      // Envoi des résultats au thread principal
      // @ts-ignore - postMessage type limitations
      postMessage({
        msg: 'lines',
        payload: result,
      } as WorkerResponse);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      // @ts-ignore - postMessage type limitations
      postMessage({
        msg: 'error',
        payload: {
          stage: 'detect',
          message: errorMessage,
          err: error,
        },
      } as WorkerResponse);
    }
  }
};

