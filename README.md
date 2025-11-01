# DBNet - Détection de Lignes de Texte dans les Images

![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?logo=vite&logoColor=white)

> **📚 Projet Personnel d'Apprentissage**
> 
> Ce projet a été développé par un étudiant à des fins d'apprentissage personnel pour comprendre l'implémentation de la détection de texte avec des modèles de deep learning dans le navigateur web. Bien qu'il ne s'agisse pas d'un travail universitaire demandé, il est traité avec le même sérieux et constitue une base solide pour des projets réels.

Application web de détection de lignes de texte dans les images en utilisant le modèle de deep learning **DBNet** (Differentiable Binarization Network). Le traitement s'effectue entièrement dans le navigateur grâce à ONNX Runtime et OpenCV.js.

## 🚀 Démo en ligne

**Lien GitHub Pages** : [Visualiser la démo](https://alpha-dklg.github.io/tp-dbnet/)

## ✨ Fonctionnalités

- 🖼️ **Détection en temps réel** : Upload d'image via sélecteur de fichier ou drag & drop
- 🔵 **Visualisation** : Affichage des boîtes de détection directement sur l'image
- 📊 **Export JSON** : Coordonnées de toutes les lignes détectées
- ⚡ **Traitement asynchrone** : Web Workers pour ne pas bloquer l'interface
- 🎯 **Précision** : Modèle DBNet optimisé pour la détection de texte

## 🛠️ Technologies utilisées

- **Vite** : Build tool moderne et rapide
- **TypeScript** : Langage typé pour une meilleure maintenabilité
- **ONNX Runtime Web** : Exécution de modèles ONNX dans le navigateur (WASM)
- **OpenCV.js** : Traitement d'images et extraction de contours (CDN)
- **Web Workers** : Calculs en arrière-plan

## 📖 À propos de DBNet

DBNet est un modèle de deep learning spécialisé dans la détection de texte dans les images. Il utilise une approche de **binarisation différenciable** qui apprend à détecter les contours de texte directement, sans étape de post-traitement complexe.

### Architecture du pipeline

1. **Préprocessing** : 
   - Redimensionnement avec conservation du ratio
   - Padding pour dimensions multiples de 32
   - Normalisation ImageNet (mean/std)
   - Conversion en tensor ONNX

2. **Inference** : 
   - Exécution du modèle DBNet
   - Génération d'une heatmap de probabilités

3. **Postprocessing** : 
   - Binarisation avec seuillage
   - Extraction de contours avec OpenCV
   - Regroupement des boîtes par lignes
   - Remise à l'échelle des coordonnées

## 📁 Structure du projet

```
tp-dbnet/
├── public/
│   └── models/
│       └── det_model.onnx      # Modèle DBNet pré-entraîné
├── src/
│   ├── main.ts                  # Interface utilisateur et événements
│   ├── postprocess.ts           # Rendu des résultats
│   └── worker/
│       └── dbnet.worker.ts      # Pipeline complet
├── index.html                   # Page principale
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## 🚦 Installation et usage

### Prérequis

- **Node.js** >= 16
- **npm** ou **yarn**

### Installation

```bash
# Cloner le repository
git clone https://github.com/alpha-dklg/tp-dbnet.git
cd tp-dbnet

# Installer les dépendances
npm install
```

### Développement local

```bash
# Lancer le serveur de développement
npm run dev
```

Ouvrez `http://localhost:5173` dans votre navigateur.

### Build pour production

```bash
# Construire l'application
npm run build

# Prévisualiser le build
npm run preview
```

Le dossier `dist/` contient l'application prête pour le déploiement.

### Déploiement sur GitHub Pages

Le projet est déjà configuré pour se déployer automatiquement sur GitHub Pages via GitHub Actions.

**Configuration** :
1. Le workflow est déjà configuré dans `.github/workflows/deploy.yml`
2. Activer GitHub Pages dans les paramètres du repository :
   - Repository → Settings → Pages
   - Source : "GitHub Actions"
3. **Push** vers `main` déclenche automatiquement le déploiement

L'application sera accessible sur : `https://alpha-dklg.github.io/tp-dbnet/`

## 📝 Utilisation

### Interface

1. **Charger une image** :
   - Cliquez sur "Choisir un fichier" ou
   - Glissez-déposez une image dans la zone dédiée

2. **Résultats** :
   - Les lignes détectées apparaissent en bleu sur l'image
   - Les coordonnées JSON s'affichent en dessous

### Format de sortie

```json
[
  { "x": 100, "y": 50, "w": 200, "h": 30 },
  { "x": 100, "y": 90, "w": 180, "h": 28 }
]
```

Chaque objet représente une ligne avec :
- `x`, `y` : Position du coin supérieur gauche
- `w`, `h` : Largeur et hauteur

## 🔧 Configuration

### Paramètres ajustables dans `src/worker/dbnet.worker.ts`

```typescript
const RESIZE_MAX_SIDE = 960;        // Taille max d'image
const THRESHOLD = 0.3;              // Seuil de probabilité
const MIN_BOX_WIDTH = 5;            // Largeur minimale des boîtes
const MIN_BOX_HEIGHT = 5;           // Hauteur minimale
const MERGE_TOL_FACTOR = 0.15;      // Tolérance de regroupement
```

## 🐛 Résolution de problèmes

**Le modèle ne se charge pas** :
- Vérifiez que `det_model.onnx` est bien dans `public/models/`
- Vérifiez la console du navigateur (F12) pour les erreurs

**Détection insuffisante** :
- Essayez d'ajuster `THRESHOLD` (plus bas = plus de détections)
- Vérifiez la résolution de l'image (images trop grandes sont redimensionnées)

**Erreur Web Worker** :
- Vérifiez que le navigateur supporte les Web Workers
- Chrome, Firefox, Edge sont compatibles

## 📚 Ressources

- [Article DBNet](https://arxiv.org/abs/1911.08947) - Paper original
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) - Documentation officielle
- [OpenCV.js](https://docs.opencv.org/4.x/d2/d00/tutorial_js_root.html) - Guide d'utilisation
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Modèles pré-entraînés

## 📖 À propos du projet

**Projet d'auto-formation et de compréhension**

Ce projet a été réalisé par un étudiant pour approfondir sa compréhension de :
- L'exécution de modèles de deep learning dans le navigateur
- Le preprocessing et postprocessing d'images
- L'utilisation de Web Workers pour des calculs intensifs
- Les technologies web modernes (Vite, TypeScript, ONNX Runtime)

**Objectifs pédagogiques** :
- ✅ Comprendre l'architecture d'un pipeline de détection de texte
- ✅ Maîtriser les concepts de preprocessing et postprocessing
- ✅ Apprendre à intégrer des modèles ONNX dans une application web
- ✅ Développer des compétences pratiques applicables à des projets réels

**Note** : Bien que ce ne soit pas un travail universitaire demandé, ce projet démontre une compréhension approfondie des concepts et peut servir de base pour des applications professionnelles.

## 🙏 Remerciements

- **DBNet** : Les auteurs du modèle original
- **PaddleOCR** : Pour les modèles pré-entraînés
- **ONNX Runtime** : Pour l'exécution dans le navigateur
- **OpenCV** : Pour le traitement d'images

---

**Auteur** : DIALLO Mamadou Alpha ([@alpha-dklg](https://github.com/alpha-dklg))

Projet développé dans le cadre d'une auto-formation personnelle pour approfondir la compréhension des concepts de deep learning dans le navigateur web.

