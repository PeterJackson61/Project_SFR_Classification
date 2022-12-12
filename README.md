# Project_SFR_Classification
### SFR failure 
* SFR: Spatial Frequency Response: Characterize the contrast of the image range(0,1). Near to 0 indicate low contrast/sharpness.
* There are 5 fields: center, 30F, 60F, 75F, edge, corresponded with the percentage of the picture. E.g. center field boundary is a circle that covers 10% of the image. 
* Pictures taken from the camera modules has 276 ROIs to be scored by SFR score. 276 ROIs correspond to 276 SFR score. Each ROIs belong to one field only.
* SFR logs contain the information about the ROIs (coordinates, SFR score).
* Picture with at least 1 ROI with the SFR score lower than lower specification limit (LSL) will be marked as failure. 
### Heatmap
* 276 SFR Score of 276 ROIs will be displayed by a color map and interpolated.
### Motivation
* Optics engineers have to control the failure rate of optical-related failure daily/weekly according to the demand
  + Define the root cause of the failure for a large amount of camera modules (particle, process tilt, lens poor performance)
  + Defining the root cause requires a long process: heatmap drawing, chart image downloading and examining, module T-focus by testers
* The tool has advantages:
  + Define the root cause only from file log. 
  + Free the engineers from waiting for T-focus long process and checking the particle in the images.
  + Arrange the heatmaps and judgements in one file that can be sent to other engineers.
  + Can be applied to the process machine for avoiding failure from process.
 ### Data preprocessing
  + Remove duplicate, sorted by time, keeping the last value of barcode (the IDs of the modules)
 ### Model chosen
 * Support vector machine
