# Object Detection IPA

This project is about detecting yellow circle shaped objects from an input image.

## Requirements
``pip3 install -r requirements.txt``

## Parameters
* ``-image_path``: The path of the input image
* ``-d_labmda``: When denoising the image we need this parameter to be greater than 0.
Default value is 1. Thus, if denoising process is not desired, this value must be set to 0.
The denoising algorithm is "Total variation denoising".
* ``-bth_type``: Method to get the threshold for the binary image. Default value es 'otsu'.
* ``-bin_th``: If `bth_type` is 'custom'. This parameter is used as the threshold instead.
* ``-erosion``: Apply erosion using a disk to the binary image.
* ``-re``: Radius of the erosion disk. Default value is 5.
* ``-dilation``: Apply dilation using a disk to the binary image.
* ``-rd``: Radius of the dilation disk. Default value is 5.
* ``-opening``: Apply opening using a disk to the binary image.
* ``-ro``: Radius of the opening disk. Default value is 5.
* ``-closing``: Apply closing using a disk to the binary image.
* ``-rc``: Radius of the closing disk. Default value is 5.
* ``-rm_ts``: Remove the small components (components with a number of pixel less than `rm_ts`)
if its value is greater than 0. Default value is 100. As in ``d_lambda`` case, set this value to 
0 if removing is not wished.
* ``-r_error_th``: Tolerated error for estimated radius when filtering circles.
* ``-output_file``: Path of the file containing the components details.

## Usage
``python main.py -image_path <image_path>``

Other parameters are optional



