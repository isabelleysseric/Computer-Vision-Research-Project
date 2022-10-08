<p align="center">
  <img src="http://vision.gel.ulaval.ca/skin/fr/images/interface/logo_lvsn.gif" />
</p>
<br/>
<br/>


# Stage de recherche en segmentation sémantique des images panoramiques
<br/>
<br/>


<p align='center'>
  <a href="https://github.com/isabelleysseric/Traffic-sign-detector">Traffic-sign-detector</a> (GitHub)
  &nbsp; • &nbsp;<a href="https://github.com/isabelleysseric/Traffic-sign-detector/wiki">Traffic-sign-detector</a> (Wiki)
  &nbsp; • &nbsp;<a href="http://vision.gel.ulaval.ca/fr/about/index.php">LVSN</a> (Laboratoire de recherche)
  &nbsp; • &nbsp;<a href="https://iid.ulaval.ca/">Institut intelligence et données (IID)</a> (Institut de recherche)
  &nbsp; • &nbsp;<a href="https://sentinellenord.ulaval.ca/projets-de-recherche/design-biophilique-dans-larctique-co-creation-communautaire">Sentinelle Nord</a> (Projet)
  &nbsp; • &nbsp;<a href="http://vision.gel.ulaval.ca/~jflalonde/students/">Groupe J-F Lalonde</a> (Equipe)
  &nbsp; • &nbsp;<a href="https://iid-ulaval.slack.com/archives/C0141TJKPH7">iid-ulaval</a> (Slack)
  &nbsp; • &nbsp;<a href="http://wcours.gel.ulaval.ca/GIF4105/index.html">Photographie Algorithmique</a> (Cours)<br/>
  
  <a href="https://github.com/isabelleysseric">isabelleysseric</a> (GitHub)
  &nbsp; • &nbsp;<a href="https://isabelleysseric.com/">isabelleysseric.com</a> (Portfolio)
  &nbsp; • &nbsp;<a href="https://www.linkedin.com/in/isabelle-eysseric/">isabelle-eysseric</a> (LinkedIn) <br/>
</p>
<br/>
<br/>


*Author: Isabelle Eysseric*
<br/>
<br/>


## Introduction

**Objectif du projet**: Generer les masques pour la segmentation semantique

<br/>
  
**Resultats**: Generer les masques pour la segmentation semantique, l'estimation de profondeur et l'estimation de position

<br/>

**Le projet s'est déroulé en 3 phases**:  

* Phase 1: Recherche et analyse de la segmentation sémantique
* Phase 2: Tests sur les deux modèles sélectionnés
* Phase 3: Generate masks
  
Dans ce référentiel, il y a 3 dossiers, un pour le code, un pour les données, un pour les images à visualiser dans le wiki et un autre pour le rapport final.
  
<br/>
<br/>
  
## Structure
  
Computer-Vision-Research-Project-main  
│  
├── code  
│      │  
│      ├── COCO  
│      │      ├── fold1  
│      │      │       └── fil1  
│      │      ├── ... 
│      │      ├── foldn  
│      │      │       └── filn 
│      │      ├── LICENSE  
│      │      └── README.md  
│      │  
│      ├── Matterport3D  
│      │      ├── fold1  
│      │      │       └── fil1  
│      │      ├── ... 
│      │      ├── foldn  
│      │      │       └── filn 
│      │      ├── LICENSE  
│      │      └── README.md  
│      │  
│      ├── Other  
│      │      ├── fold1  
│      │      │       └── fil1  
│      │      ├── ... 
│      │      ├── foldn  
│      │      │       └── filn 
│      │      ├── LICENSE  
│      │      └── README.md  
│      │  
│      └── README.md  
│     
├── data  
│      │  
│      ├── COCO_detection  
│      │      ├── image_1.png  
│      │      ├── ...    
│      │      ├── image_n.png
│      │      └── README.md  
│      │  
│      ├── Matterport3D  
│      │      ├── image_1.png  
│      │      ├── ...    
│      │      ├── image_n.png
│      │      └── README.md  
│      │ 
│      ├── stanford2D  
│      │      ├── image_1.png  
│      │      ├── ...    
│      │      ├── image_n.png
│      │      └── README.md  
│      │  
│      └── README.md  
│     
├── images  
│      │  
│      ├── image_1.png  
│      ├── ...  
│      └── image_n.png  
|  
├── rapport  
│      ├── Rapport_final_de_stage.pdf  
│      └── README.md  
│  
└── README.md 

<br/>
<br/>
  
