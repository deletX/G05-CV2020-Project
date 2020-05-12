# Appunti

Roba interessante dai papers degli studenti dell'anno scorso

Index: 
* [01.pdf](#01pdf)
* [02.pdf](#02pdf)
* [03.pdf](#03pdf)
* [04.pdf](#04pdf)
* [05.pdf](#05pdf)
* [06.pdf](#06pdf)

- [Note](#Note)

## 01.pdf
Provano sia con le tecniche di Image Processing che con CNNs.
### Image Processing
Usano la seguente pipeline:
1. **Ridurre la dimensione del video in lettura**: lo fanno per migliorare i risultati e le prestazioni
1. **HSV ed applicazione di una soglia**: La soglia la scelgono prendendo diversi valori da diverse immagini e facendo una media (meh)
1. **Doppia Dilazione**: per rimuovere il rumore dalla maschera (tipo le cose dentro ai quadri, sono sottili e spariscono con la dilazione)
1. **Smooth the edges**: applicano un median filter `cv.medianBlur()`
1. **find countours and draw Roi Rectangle**: Trovano i contorni come punti continui tra i confini delle regioni (forse aveva più senso continuare con una segmentazione)
1. **Recognize the painting**: Seguono la pipeline standard per la retrieval, hanno un db coi sift descriptors per ogni painting. (hanno una interfaccia, menzionano un doppio click). Usano un _FLANN matcher_.
1. **Rectify the painting**

#### Assunzioni
Loro assumono che un quadro abbia un area > di 6000 suared pixels e che i muri abbiano lo stesso colore su tutta l'immagine.
### CNNs
Usano una rete per dire quadro/not quadro, non interessante.

## 02.pdf
Dividono il loro lavoro in:
 1. **Preprocessing**
 1. **Detection and Segmentation**
 1. **Rectification**
 
### Preprocessing
Si compone di due step:
1. **Distortion Correction**: Utilizzano la soluzione della cucchiara: [”A Hough Transform-based method
for Radial Lens Distortion Correction” by R. Cucchiara, C.
Grana, A. Prati, R. Vezzani](https://aimagelab.ing.unimore.it/imagelab/pubblicazioni/iciap2003_Hough.pdf) (è un algoritmo iterativo che sostanzialmente applica Canny e la Hough Transform e determina quando le _edges_ siano dritte). Lo impostano come opzionale
1. **Noise and Blurr Filtering**: Anche questi resizano l'immagine ad una dimensione fissata. Applicano un filtro gaussiano ed altri _simple normalization steps_. Aggiustano la brigthness se l'immagine è troppo scura.

### Detection and Segmentation
Anche questi applicano una soglia (Otsu) per ottenere le aree che migliorano facendo erosioni e dilazioni ed evntualmente riempiendo delle aree coi buchi.
Applicano dei limiti per evitare incovenienti:
*   Usano un valore minimo per altezza e larghezza (50)
*   L'area del contorno dell'oggetto dovrebbe essere almeno il 60% dell'area del rettangolo (?)
*   La larghezza della bbox non dev'essere inferiore del 40% dell'altezza e viceversa (per rimuovere elementi troppo sottili o stretti)
*   il valore assoluto della differenza della media dei colori della sotto immagine nel bbox e di quella del background dev'essere superiore ad una soglia

### Rectification
Usano canny per fare edge detection, trovano i punti più lontani dal centro  ed usano come punti destinazione gli angoli della bbox (bad)

## 03.pdf
Usano la Image processing.

Pipeline:
* Median filter: per rimuovere il rumore
* modificano contrasto e luminance
* convertono l'immagini in una scala di grigi ed usano una *adaptive threshold* per identificare gli oggetti
* Applicano _dilation_ e _closing_ per rimuovere rumore e piccoli buchi nelle shapes.
* Trovano i contorni usando `.findContours` e riempono l'interno con `.drawContours`. (Solo per un area > di 120x120 pixels)
* Usano `.connectedComponentsWithStats` che fa il labeling delle varie shape e fornisce informazioni aggiuntive (area, centroide, altezza e larghezza)
* Usano un **Harris detector** per trovare gli angoli e poi fanno la rectification usando la bbox (bleah)

## 04.pdf
Usano la segmentazione. La loro pipeline:
* Segmentazione
    * Resizing e stretch contrast: 
        * Sto gaussian stretching rende i colori più chiari più chari ed i più scuri più scuri.
    * smoothing: un filtro gaussiano con sigmaX ed Y calcolate automaticametne.
        * Usano `.PyrMeanShilftFiltering` per eliminare _perceuptual noise_
    * Thresholding: applicano otsu e poi una _opening_ (che sono una erosione ed una dilazione in successione)
* Detection: usano `.findCountours()` per trovare i contorni e `.getRectCOord(contour)` ed altre funzione per ottenere la bbox. Sulla quale applicano delle assunzioni per eliminare oggetti indesiderati (interessante, oltre alle solite assunzione su area e rapporto altezza/larghezza è l'istogramma del contenuto che assumono sia più scuro dello sfondo(interessante, ma troppo sbatti))
* Rectification: Anche questi rettificano prendendo gli angoli del quadro e portandoli a quelli della bbox.

Anche questo gruppo usa la tecnica per rimoazione della _lens distortion_ della cuchciara come nel [02.pdf](#02pdf)

## 05.pdf

Dividono il lavoro in 
* Camera Correction: usano anche qui il coso della cucchiara (che a quanto pare prende da un lavoro di [Devernay e Faugeras](https://hal.inria.fr/inria-00267247/file/distcalib.pdf) ?)
* Segmentazione: ne applicano due contemporaneamente.
    * A (che dovrebbe intercettare elementi più grandi)
        * soglia sulla birghtness. 
        * Rendono l'immagine binaria separando i quadri/oggetti dal background (muro) attraverso un k-means con k=2 => 2-means. Lo sfondo viene convertito a nero ed il foreground in bianco. 
        * Per migliorare la selezione fanno una dilazione.
        * Labeling con 8-way connectivity secondo un [paper di Grana](https://aimagelab.ing.unimore.it/imagelab/pubblicazioni/2009TIPlabeling.pdf)
        * applicano un filtro sulla dimensione e __poi__ applicano un closing
        * vengono identificati i contorni (con un algoritmo dell'83, di OpenCV)
        * fanno un lavoro strano ed una erosione ? Non ho ben capito qui
    * B (che dovrebbe intercettare elementi più piccoli)
        * convertono in grigio, applicano filtro bilaterale ed una __adaptive threshold__ con un filtro gaussano

* Rettificazione: usano un modo molto interessante. Trovano le edges con canny  e una hough transform probabilistica e stimano i __vanishing point__ con un modello RANSAC

## 06.pdf
(Oh mio dio i related works che palle)

Che schifo.

Di interessante c'è che usano diverse tecniche di thresholding per poi unirle per migliorare il risultato.

Applicano Harris per trovare gli angoli e li estendono alla bbox (bleah)

(n mezzo ci sono sempre openings, filtri gaussiani etc.)

## 07.pdf
Usano la tecnica della cucchiara per fare la camera correction e la tecnica di Grana per fare labelling.

Applicano gauyssian filter ed una adaptive threshold. 

Usano una tecnica interessante per fare la rettificaizone

## Note
Quando _tweakiamo_ degli iperparametri, (costnati di regolarizzazione, dimensione di vicinati, vattelapesca, etc...) ricordiamoci di creare dei grafici che provino i valori che abbiamo scelto effettivamente siano i milgiori secondo una determinata misura. 

Sono belli da mettere e danno autorevolezza al risultato.

Interessante dire quello che usa OpenCV invece che dire abbiamo usato questa funzione di openCV