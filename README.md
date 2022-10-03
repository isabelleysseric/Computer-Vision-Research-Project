<p align="center">
  <img src="http://vision.gel.ulaval.ca/skin/fr/images/interface/logo_lvsn.gif" />
</p>
<br/>
<br/>


# Research internship in semantic segmentation for panoramic images.
<br/>
<br/>


&nbsp; • &nbsp;[LVSN](http://vision.gel.ulaval.ca/fr/about/index.php) (Laboratoire de recherche)
&nbsp; • &nbsp;[Institut intelligence et données (IID)](https://iid.ulaval.ca/) (Institut de recherche)
&nbsp; • &nbsp;[Sentinelle Nord](https://sentinellenord.ulaval.ca/projets-de-recherche/design-biophilique-dans-larctique-co-creation-communautaire) (Projet)
&nbsp; • &nbsp;[Groupe J-F Lalonde](http://vision.gel.ulaval.ca/~jflalonde/students/) (Equipe)
&nbsp; • &nbsp;[iid-ulaval](https://iid-ulaval.slack.com/archives/C0141TJKPH7) (Slack)
&nbsp; • &nbsp;[Photographie Algorithmique](http://wcours.gel.ulaval.ca/GIF4105/index.html) (Cours)
&nbsp; • &nbsp;[Computer-Vision-Research-Project](https://github.com/isabelleysseric/Computer-Vision-Research-Project) (GitHub)
&nbsp; • &nbsp;[Computer-Vision-Research-Project](https://github.com/isabelleysseric/Computer-Vision-Research-Project/wiki) (Wiki)
&nbsp; • &nbsp;[isabelleysseric](https://hub.docker.com/u/isabelleysseric) (Docker)
&nbsp; • &nbsp;[isabelleysseric.com](https://isabelleysseric.com) (Portfolio)
&nbsp; • &nbsp;[isabelle-eysseric](https://www.linkedin.com/in/isabelle-eysseric/) (Linkedin)
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
  
