# TP Détection de Lignes de Texte avec DBNet

## Objectif pédagogique

L'objectif de ce TP est de comprendre et implémenter un système de détection de lignes de texte sur des images en utilisant un modèle de deep learning (DBNet) optimisé pour le navigateur web. Vous allez travailler avec les technologies web modernes, le traitement d'images en temps réel, et l'inference de modèles ONNX.

## Contexte théorique

### DBNet : Différenciable Binarisation Network

DBNet est un modèle de deep learning spécialisé dans la détection de texte dans les images. Contrairement aux méthodes traditionnelles de binarisation (transformation d'une image en noir et blanc), DBNet apprend à détecter les contours de texte directement sans étape de post-traitement complexe.

**Architecture** :
- **Input** : Image RGB normalisée
- **Backbone** : Réseau de convolution (ex: ResNet) pour extraire les caractéristiques
- **Neck** : Feature Pyramid Network (FPN) pour fusionner les features multi-échelles
- **Head** : Génération de probabilités pour chaque pixel (text/non-text)
- **Output** : Heatmap de probabilités

### Pipeline de traitement

Le système se compose de trois étapes principales :

1. **Préprocessing** : Adaptation de l'image pour le modèle
2. **Inference** : Exécution du modèle ONNX
3. **Postprocessing** : Transformation de la heatmap en boîtes de texte

## Structure du projet

```
tp-dbnet/
├── public/
│   └── models/
│       └── det_model.onnx      # Modèle DBNet pré-entraîné
├── src/
│   ├── main.ts                  # Interface utilisateur et gestion des événements
│   ├── postprocess.ts           # Rendu des résultats
│   └── worker/
│       └── dbnet.worker.ts      # Pipeline complet (préprocessing, inference, postprocessing)
├── index.html                   # Interface web
├── package.json                 # Dépendances du projet
└── tsconfig.json                # Configuration TypeScript
```

## Technologies utilisées

- **Vite** : Build tool moderne pour applications web
- **TypeScript** : Langage typé pour JavaScript
- **ONNX Runtime Web** : Exécution de modèles ONNX dans le navigateur
- **OpenCV.js** : Bibliothèque de traitement d'images (chargée via CDN)
- **Web Workers** : Traitement parallèle pour ne pas bloquer l'interface

## Instructions de mise en place

### 1. Installation des dépendances

```bash
npm install
```

### 2. Lancement de l'application

```bash
npm run dev
```

L'application sera accessible sur `http://localhost:5173`

## Exercices pratiques

### Partie 1 : Comprendre le préprocessing (30 min)

**Objectif** : Analyser et documenter l'étape de préprocessing.

**Consignes** :
1. Ouvrez le fichier `src/worker/dbnet.worker.ts`
2. Étudiez la fonction `preprocess()` (lignes 185-228)
3. Répondez aux questions suivantes :

   a) **Resize** : Pourquoi redimensionnons-nous l'image avec `RESIZE_MAX_SIDE = 960` ?
   
   b) **Padding** : Pourquoi ajoutons-nous du padding pour avoir des dimensions multiples de 32 ?
   
   c) **Normalisation** : 
      - Pourquoi divisons-nous par 255 les valeurs RGB ?
      - À quoi correspondent les constantes `NORMALIZATION_MEAN` et `NORMALIZATION_STD` ?
   
   d) **Format du tensor** : Le tensor final est de format `[1, 3, H, W]`. Expliquez chaque dimension.

**Exercice pratique** : Modifiez la constante `RESIZE_MAX_SIDE` à 480 pixels. Observez l'impact sur :
- La rapidité de traitement
- La qualité de détection (précision des boîtes)
- La mémoire utilisée (inspectez dans les DevTools)

Documentez vos observations.

### Partie 2 : Analyser l'inference (20 min)

**Objectif** : Comprendre l'exécution du modèle ONNX.

**Consignes** :
1. Étudiez la fonction `runInference()` (lignes 251-297)
2. Répondez aux questions :

   a) Pourquoi le modèle est-il chargé une seule fois et mis en cache ?
   
   b) Le modèle retourne une heatmap de probabilités. Expliquez ce concept.
   
   c) Pourquoi avons-nous un fallback avec `images` comme nom d'entrée ?

**Note** : `onmessage` (ligne 453) est le point d'entrée du Web Worker. C'est ici que le pipeline complet est orchestré.

### Partie 3 : Maîtriser le postprocessing (40 min)

**Objectif** : Implémenter et améliorer la détection des lignes de texte.

**Consignes** :
1. Étudiez la fonction `postprocess()` (lignes 338-444)
2. Tracez les étapes suivantes :

   a) **Binarisation** : 
      - Pourquoi multiplions-nous la heatmap par 255 ?
      - Que fait `cv.threshold()` avec `THRESH_BINARY` ?
      - Quel impact a la constante `THRESHOLD = 0.3` ?

   b) **Extraction des contours** :
      - À quoi sert `cv.findContours()` ?
      - Pourquoi filtrons-nous les boîtes avec `MIN_BOX_WIDTH` et `MIN_BOX_HEIGHT` ?

   c) **Regroupement par lignes** :
      - Expliquez l'algorithme de regroupement (lignes 398-434)
      - Comment est calculée la `verticalTolerance` ?
      - Pourquoi trie-t-on les boîtes par centre vertical puis par position horizontale ?

**Exercice pratique 1** : Modifiez le seuil `THRESHOLD` :
- Testez avec 0.1, 0.2, 0.4, 0.5
- Documentez pour chaque valeur :
  - Nombre de fausses détections
  - Nombre de détections manquées
  - Cohérence visuelle des boîtes

**Exercice pratique 2** : Implémentez un filtrage supplémentaire :
- Ajoutez un filtre qui supprime les boîtes avec un ratio largeur/hauteur > 10
- Justifiez pourquoi ce filtre est pertinent
- Testez sur plusieurs images

**Exercice pratique 3** : Améliorez le regroupement par lignes :
- Actuellement, `verticalTolerance` est calculée comme 15% de la hauteur moyenne
- Testez des valeurs de 10%, 20%, 25%
- Évaluez l'impact sur des images avec des lignes irrégulières

### Partie 4 : Interface utilisateur (20 min)

**Objectif** : Interagir avec l'application et visualiser les résultats.

**Consignes** :
1. Étudiez `src/main.ts`
2. Comprenez le flux de données :
   - Comment l'image est-elle chargée ?
   - Comment est-elle envoyée au Web Worker ?
   - Comment les résultats sont-ils affichés ?

3. Testez l'application avec différentes images :
   - Image avec texte horizontal
   - Image avec texte incliné
   - Image avec plusieurs paragraphes
   - Image avec du texte sur fond complexe

**Exercice pratique** : Ajoutez des statistiques :
- Nombre de lignes détectées
- Temps de traitement (depuis l'envoi au worker jusqu'à la réception)
- Dimensions moyennes des lignes
- Affichez ces statistiques sous le JSON dans l'interface

### Partie 5 : Optimisations et défis (bonus)

**Challenge 1** : Gestion des images de grande taille
- Actuellement, `RESIZE_MAX_SIDE = 960` peut être limitant pour très grandes images
- Implémentez un redimensionnement adaptatif qui :
  - Conserve les petites images (< 960px) à leur taille originale
  - Redimensionne les grandes images tout en conservant le ratio
  - Testez sur des images de 2000x3000 pixels

**Challenge 2** : Gestion du texte incliné
- DBNet détecte bien le texte horizontal
- Pour le texte incliné, il crée des boîtes englobantes rectangulaires
- Proposez une méthode pour détecter l'angle d'inclinaison et roter les boîtes

**Challenge 3** : WebGL backend
- Actuellement, ONNX Runtime utilise WASM
- CONFIGURE l'application pour utiliser WebGL (GPU)
  ```typescript
  executionProviders: ['webgl', 'wasm']
  ```
- Mesurez l'amélioration de performance

**Challenge 4** : Export des résultats
- Implémentez un bouton pour :
  - Exporter les images croppées (une par ligne détectée)
  - Exporter les coordonnées au format JSON, XML, ou CSV
  - Sauvegarder l'image originale avec les boîtes dessinées

## Questions de synthèse

1. **Architecture** : Pourquoi utilisons-nous un Web Worker plutôt que le thread principal pour l'inference ?

2. **Normalisation** : Les valeurs de normalisation ImageNet sont souvent utilisées dans les modèles pré-entraînés. Expliquez pourquoi.

3. **Contraintes DBNet** : Pourquoi les dimensions doivent-elles être multiples de 32 ? (Indice : pensez aux opérations de convolution et pooling)

4. **Complexité** : Calculez la complexité temporelle du postprocessing :
   - O(n) pour l'extraction des contours (n = nombre de pixels)
   - O(m log m) pour le tri (m = nombre de boîtes)
   - O(m) pour le regroupement

5. **Robustesse** : Identifiez au moins 3 cas limites où le système pourrait échouer et proposez des solutions.

## Critères d'évaluation

- **Compréhension théorique** (30%) : Réponses précises aux questions
- **Code et implémentation** (30%) : Modifications fonctionnelles et bien documentées
- **Tests et observations** (20%) : Expérimentations rigoureuses avec documentation
- **Bonus** (20%) : Challenges implémentés et fonctionnels

## Ressources complémentaires

- [Article original DBNet](https://arxiv.org/abs/1911.08947)
- [Documentation ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)
- [OpenCV.js Reference](https://docs.opencv.org/4.x/d2/d00/tutorial_js_root.html)
- [Web Workers API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API)

## Durée estimée

- Partie 1 : 30 minutes
- Partie 2 : 20 minutes
- Partie 3 : 40 minutes
- Partie 4 : 20 minutes
- Questions de synthèse : 20 minutes
- Bonus : variable

**Total** : ~2h30 sans les bonus

## Remise du travail

1. Code commenté avec vos modifications
2. Document PDF/Word contenant :
   - Réponses aux questions
   - Observations des exercices pratiques
   - Captures d'écran des tests
   - Résultats des challenges (si complétés)
3. Archive ZIP avec l'ensemble du projet

**Note** : Le code doit compiler sans erreur et être exécutable avec `npm run dev`.

---

**Bonne chance dans votre apprentissage de la détection de texte !** 📝🤖

