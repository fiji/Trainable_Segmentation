[![](https://github.com/fiji/Trainable_Segmentation/actions/workflows/build-main.yml/badge.svg)](https://github.com/fiji/Trainable_Segmentation/actions/workflows/build-main.yml)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.59290.svg)](http://dx.doi.org/10.5281/zenodo.59290)

Trainable Weka Segmentation
======================
The **Trainable Weka Segmentation** is a Fiji plugin and library that combines a collection of machine learning algorithms with a set of selected image features to produce pixel-based segmentations. [Weka (Waikato Environment for Knowledge Analysis)](http://www.cs.waikato.ac.nz/ml/weka/) can itself be called from the plugin. It contains a collection of visualization tools and algorithms for data analysis and predictive modeling, together with graphical user interfaces for easy access to this functionality. As described on their wikipedia site, the advantages of Weka include:

- freely availability under the GNU General Public License
- portability, since it is fully implemented in the Java programming language and thus runs on almost any modern computing platform
- a comprehensive collection of data preprocessing and modeling techniques
- ease of use due to its graphical user interfaces
- Weka supports several standard data mining tasks, more specifically, data preprocessing, clustering, classification, regression, visualization, and feature selection.

The main goal of this library is to work as a **bridge between the Machine Learning and the Image Processing fields**. It provides the framework to use and, more important, compare any available classifier to perform image segmentation based on pixel classification.

For further details, please visit the [documentation site](https://imagej.net/Trainable_Weka_Segmentation).

![Trainable Weka Segmentation pipeline overview](https://imagej.net/media/plugins/tws/tws-pipeline.png)

Citation
--------
Please note that Trainable Weka Segmentation is based on a publication. If you use it successfully for your research please be so kind to cite our work:
* Arganda-Carreras, I., Kaynig, V., Rueden, C., Eliceiri, K. W., Schindelin, J., Cardona, A., & Seung, H. S. (2017). [Trainable Weka Segmentation: a machine learning tool for microscopy pixel classification](https://academic.oup.com/bioinformatics/article-abstract/doi/10.1093/bioinformatics/btx180/3092362/Trainable-Weka-Segmentation-a-machine-learning). Bioinformatics (Oxford Univ Press) 33 (15), [doi:10.1093/bioinformatics/btx180](http://dx.doi.org/10.1093%2Fbioinformatics%2Fbtx180) ([on Google Scholar](http://scholar.google.com/scholar?cluster=12995971888361615836)).
